import os
import argparse
import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod  # Added for Abstract Base Class
from typing import List, Tuple, Dict, Any, Union, Type  # Added Type

# Assuming these imports exist in the project structure
from .DataModule import fetch_data_module, DATA_MODULE_REGISTRY
from .NetworkModule import (
    NETWORK_REGISTRY,
    get_network,
)  # Assuming NetList is also accessible if needed by fallback
from .logging_utils import setup_logging  # Assuming this helper exists
from .vis import show_image, compute_norm_tensor_list, sum_norm

import warnings
import logging
import gc
import json

# --- Standard Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
warnings.simplefilter(action="ignore", category=FutureWarning)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants ---
BATCH_SIZE_ICML = 200
LR_ICML = 0.01
MOMENTUM_ICML = 0.9
NUM_EPOCHS_ICML = 100

# --- Unified Helper Functions (Keep these as they are generally useful) ---


def get_device(gpu: int) -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return f"cuda:{gpu}"
    else:
        return "cpu"


def get_file_paths(
    key: str,
    model_type: str,
    seed: int,
    infl_type: str = None,
    save_dir: str = None,
    relabel_percentage: float = None,
) -> Tuple[str, str, str]:
    """
    Generates directory, fallback data path, and influence output path.

    Returns:
        Tuple[str, str, str]: dn (directory), fn (fallback .dat path), gn (influence .dat path)
    """
    base_dir = (
        os.path.join(SCRIPT_DIR, save_dir)
        if save_dir
        else os.path.join(SCRIPT_DIR, f"{key}_{model_type}")
    )
    os.makedirs(base_dir, exist_ok=True)  # Ensure directory exists

    relabel_prefix = (
        f"relabel_{int(relabel_percentage):03d}_pct_"
        if relabel_percentage is not None
        else ""
    )
    seed_suffix = f"{seed:03d}.dat"
    json_seed_suffix = f"{seed:03d}.json"
    csv_seed_suffix = f"{seed:03d}.csv"

    # Fallback/Main results file (.dat)
    sgd_prefix = "sgd_" if relabel_percentage is None else ""  # Assuming convention
    fn = os.path.join(base_dir, f"{relabel_prefix}{sgd_prefix}{seed_suffix}")
    # Fallback JSON name
    fn_json_fallback = os.path.join(
        base_dir, f"{relabel_prefix}global_info_{json_seed_suffix}"
    )
    fn_json_no_relabel = os.path.join(base_dir, f"global_info_{json_seed_suffix}")

    # Influence output file path (.dat)
    gn = ""
    if infl_type:
        # Handle special full paths first
        if infl_type == "lie_full":
            gn = os.path.join(base_dir, f"infl_lie_full_{relabel_prefix}{seed_suffix}")
        elif infl_type == "segment_true_full":
            gn = os.path.join(
                base_dir, f"infl_segment_true_full_{relabel_prefix}{seed_suffix}"
            )
        # Standard influence file naming
        else:
            gn = os.path.join(
                base_dir, f"infl_{infl_type}_{relabel_prefix}{seed_suffix}"
            )

    return base_dir, fn, gn


def load_global_info(
    dn: str,
    seed: int,
    fn_fallback_dat: str,
    device: str,
    logger: logging.Logger,
    relabel_percentage: float = None,
) -> Dict[str, Any]:
    """Loads global training info from JSON, falling back to .dat file."""
    relabel_prefix = (
        f"relabel_{int(relabel_percentage):03d}_pct_"
        if relabel_percentage is not None
        else ""
    )
    json_fn_base = f"global_info_{seed:03d}.json"
    json_fn_relabel = f"{relabel_prefix}{json_fn_base}"

    json_paths_to_try = (
        [
            os.path.join(dn, json_fn_relabel),  # Try relabeled name first if applicable
            os.path.join(dn, json_fn_base),  # Then try standard name
        ]
        if relabel_percentage is not None
        else [os.path.join(dn, json_fn_base)]
    )

    loaded_from_json = False
    for json_fn in json_paths_to_try:
        try:
            if os.path.exists(json_fn):
                with open(json_fn, "r") as f:
                    global_info = json.load(f)
                logger.info(
                    f"Successfully loaded global information from JSON: {json_fn}"
                )
                loaded_from_json = True
                # Basic validation of loaded JSON
                required_keys = [
                    "n_tr",
                    "n_val",
                    "n_test",
                    "num_epoch",
                    "batch_size",
                    "lr",
                    "alpha",
                ]  # 'decay' might be optional or named 'weight_decay'
                missing_keys = [
                    k
                    for k in required_keys
                    if k not in global_info or global_info.get(k) is None
                ]
                if missing_keys:
                    logger.warning(
                        f"Loaded JSON {json_fn} is missing keys: {missing_keys}. Fallback might occur."
                    )
                    # Decide if this is critical enough to force fallback
                    # loaded_from_json = False # Uncomment to force fallback if keys missing
                else:
                    # Add decay if missing but weight_decay exists
                    if (
                        global_info.get("decay") is None
                        and global_info.get("weight_decay") is not None
                    ):
                        global_info["decay"] = global_info["weight_decay"]
                        logger.info("Mapped 'weight_decay' to 'decay' from JSON.")
                    elif global_info.get("decay") is None:
                        global_info["decay"] = 0.0  # Assume 0 if not present
                        logger.info("Assuming 'decay'=0.0 as it was not found in JSON.")
                    return global_info  # Success
        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding JSON file {json_fn}: {e}. Trying next.")
        except Exception as e:
            logger.warning(f"Error reading JSON file {json_fn}: {e}. Trying next.")

    # --- Fallback to .dat ---
    if not loaded_from_json:
        logger.warning(
            f"Global info JSON not found or invalid (tried: {json_paths_to_try}). Falling back to .dat file: {fn_fallback_dat}"
        )
        try:
            res = torch.load(fn_fallback_dat, map_location=device, weights_only=False)
            # Reconstruct global_info, checking for key existence
            global_info = {
                "seed": res.get("seed", seed),
                "n_tr": res.get("n_tr"),
                "n_val": res.get("n_val"),
                "n_test": res.get("n_test"),
                "num_epoch": res.get("num_epoch"),
                "batch_size": res.get("batch_size"),
                "lr": res.get("lr"),
                "decay": res.get("decay", res.get("weight_decay")),  # Try both names
                "alpha": res.get("alpha"),  # Regularization param
            }
            required_keys = [
                "n_tr",
                "n_val",
                "n_test",
                "num_epoch",
                "batch_size",
                "lr",
                "alpha",
            ]  # Decay optional
            missing_keys = [k for k in required_keys if global_info.get(k) is None]
            if missing_keys:
                logger.error(
                    f"Fallback file {fn_fallback_dat} is missing required keys for global_info: {missing_keys}"
                )
                raise ValueError(
                    f"Fallback file {fn_fallback_dat} missing keys: {missing_keys}"
                )

            if global_info.get("decay") is None:
                global_info["decay"] = 0.0  # Assume 0 if not found in fallback
                logger.info(
                    "Assuming 'decay'=0.0 as it was not found in fallback .dat."
                )

            logger.info(
                f"Successfully reconstructed global information from fallback file: {fn_fallback_dat}"
            )
            return global_info
        except FileNotFoundError:
            logger.error(f"Fallback file {fn_fallback_dat} also not found.")
            raise FileNotFoundError(
                f"Could not load global training information from JSON or fallback {fn_fallback_dat}"
            )
        except Exception as e:
            logger.error(
                f"Error loading or parsing fallback file {fn_fallback_dat}: {e}"
            )
            raise


def load_data(
    key: str,
    global_info: Dict[str, Any],
    seed: int,
    device: str,
    logger: logging.Logger,
    relabel_percentage: float = None,
    dn: str = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loads and prepares training and validation data."""
    n_tr = global_info["n_tr"]
    n_val = global_info["n_val"]
    n_test = global_info[
        "n_test"
    ]  # Although often unused here, load it for consistency

    module = fetch_data_module(
        key, data_dir=os.path.join(SCRIPT_DIR, "data"), logger=logger, seed=seed
    )
    module.append_one = False  # Specific to this project's DataModule?

    z_tr, z_val, z_test = module.fetch(n_tr, n_val, n_test, seed)
    (x_tr_np, y_tr_np), (x_val_np, y_val_np) = z_tr, z_val
    # (x_test_np, y_test_np) = z_test # Keep test data if needed later

    # Convert to tensor
    x_tr = torch.from_numpy(x_tr_np).to(torch.float32).to(device)
    y_tr = torch.from_numpy(y_tr_np).to(torch.float32).unsqueeze(1).to(device)
    x_val = torch.from_numpy(x_val_np).to(torch.float32).to(device)
    y_val = torch.from_numpy(y_val_np).to(torch.float32).unsqueeze(1).to(device)
    # x_test = torch.from_numpy(x_test_np).to(torch.float32).to(device)
    # y_test = torch.from_numpy(y_test_np).to(torch.float32).unsqueeze(1).to(device)

    if relabel_percentage and dn:
        relabel_prefix = f"relabel_{int(relabel_percentage):03d}_pct_"
        idx_csv_name = os.path.join(
            dn, f"{relabel_prefix}indices_{seed:03d}.csv"
        )  # Standardized name
        logger.info(
            f"Attempting to relabel {relabel_percentage}% of training data using {idx_csv_name}"
        )
        try:
            relabel_indices_df = pd.read_csv(idx_csv_name)
            relabel_col = None
            possible_cols = ["relabel_indices", "index", "idx"]
            for col in possible_cols:
                if col in relabel_indices_df.columns:
                    relabel_col = col
                    break
            if relabel_col is None:
                raise ValueError(
                    f"Cannot find relabeled indices column in {idx_csv_name} (tried: {possible_cols})"
                )

            relabel_indices = relabel_indices_df[relabel_col].values

            if len(relabel_indices) > 0:
                max_idx = relabel_indices.max()
                if max_idx >= n_tr:
                    logger.error(
                        f"Relabeled index {max_idx} is out of bounds for training data size {n_tr}. Check {idx_csv_name}"
                    )
                    raise IndexError("Relabeled index out of bounds")

                y_tr[relabel_indices] = 1 - y_tr[relabel_indices]
                logger.info(f"Successfully relabeled {len(relabel_indices)} samples.")
            else:
                logger.warning(
                    f"Relabeled indices file {idx_csv_name} was empty or contained no indices."
                )

        except FileNotFoundError:
            logger.error(
                f"Relabeled indices file not found: {idx_csv_name}. Training data NOT relabeled."
            )
            # Decide if this is critical
            # raise # Uncomment to stop if relabeling is mandatory
        except Exception as e:
            logger.error(
                f"Error reading or applying relabeled indices from {idx_csv_name}: {e}. Training data NOT relabeled."
            )
            # raise # Uncomment to stop if relabeling is mandatory

    return x_tr, y_tr, x_val, y_val  # Return only train/val as standard


def get_input_dim(x: torch.Tensor, model_type: str) -> Union[int, Tuple[int, ...]]:
    """Determines input dimension based on data shape and model type."""
    if model_type == "cnn":
        if x.dim() == 4:  # Batch, Channel, H, W
            return x.shape[1:]  # C, H, W
        elif x.dim() == 3:  # Assuming Batch, H, W (needs channel dim)
            img_size = x.shape[1]
            # Assuming grayscale, add channel dimension
            return (1, img_size, img_size)  # C=1, H, W
        elif x.dim() == 2:  # Batch, Features (Flattened image)
            num_features = x.shape[1]
            img_size = int(np.sqrt(num_features))
            if img_size * img_size != num_features:
                raise ValueError(
                    f"Cannot infer image dimensions for CNN from flattened input of size {num_features}"
                )
            # Assuming grayscale
            return (1, img_size, img_size)  # C=1, H, W
        else:
            raise ValueError(f"Unsupported input dimension for CNN: {x.dim()}")
    elif model_type in ["logreg", "dnn"]:  # Models expecting flat features
        if x.dim() > 2:  # E.g., image data passed to MLP
            return x.shape[1:].numel()  # Flatten dimensions after batch
        else:  # Already Batch, Features
            return x.shape[1]
    else:  # Default: number of features
        if x.dim() > 2:
            # Assuming needs flattening
            return x.shape[1:].numel()
        else:
            return x.shape[1]


def compute_gradient(
    x: torch.Tensor, y: torch.Tensor, model: torch.nn.Module, loss_fn: torch.nn.Module
) -> List[torch.Tensor]:
    """Computes the gradient of the loss w.r.t. model parameters (NO regularization)."""
    model.eval()  # Ensure model is in eval mode for consistent gradient computation
    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    # Note: Regularization term (e.g., L2) is NOT included here.
    loss.backward()
    grads = [
        (p.grad.data.clone() if p.grad is not None else torch.zeros_like(p))
        for p in model.parameters()
    ]
    # Detach gradients (already done by .data, but explicit is fine)
    # for g in grads:
    #     g.requires_grad = False
    model.zero_grad()  # Clean up gradients
    return grads


def load_epoch_data(
    dn: str,
    epoch: int,
    seed: int,
    relabel_percentage: float = None,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Loads saved data for a specific epoch, handling relabeling."""
    records_dir = os.path.join(dn, "records")
    relabel_prefix = (
        f"relabel_{int(relabel_percentage):03d}_pct_"
        if relabel_percentage is not None
        else ""
    )
    epoch_file_name = f"{relabel_prefix}epoch_{epoch}_{seed:03d}.pt"
    epoch_file = os.path.join(records_dir, epoch_file_name)

    if logger:
        logger.debug(f"Attempting to load epoch file: {epoch_file}")
        logger.debug(
            f"Epoch file details - Directory: {records_dir}, Prefix: {relabel_prefix}, Epoch: {epoch}, Seed: {seed:03d}"
        )

    if not os.path.exists(epoch_file):
        # Fallback: try without relabel prefix if relabel was specified but file not found
        if relabel_percentage is not None:
            epoch_file_no_relabel = os.path.join(
                records_dir, f"epoch_{epoch}_{seed:03d}.pt"
            )
            if os.path.exists(epoch_file_no_relabel):
                if logger:
                    logger.debug(
                        f"Using non-relabel epoch file as fallback: {epoch_file_no_relabel}"
                    )
                epoch_file = epoch_file_no_relabel
            else:
                if logger:
                    logger.debug(
                        f"Fallback epoch file also not found: {epoch_file_no_relabel}"
                    )
                raise FileNotFoundError(
                    f"Could not find epoch file for epoch {epoch}, seed {seed} (tried w/ and w/o relabel prefix at {records_dir})"
                )
        else:
            raise FileNotFoundError(f"Could not find epoch file: {epoch_file}")

    return torch.load(epoch_file)  # Consider map_location=device


def load_step_data(
    dn: str,
    step: int,
    seed: int,
    relabel_percentage: float = None,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Loads saved data for a specific step, handling relabeling."""
    records_dir = os.path.join(dn, "records")
    relabel_prefix = (
        f"relabel_{int(relabel_percentage):03d}_pct_"
        if relabel_percentage is not None
        else ""
    )
    step_file_name = f"{relabel_prefix}step_{step}_{seed:03d}.pt"
    step_file = os.path.join(records_dir, step_file_name)

    if logger:
        logger.debug(f"Attempting to load step file: {step_file}")
        logger.debug(
            f"Step file details - Directory: {records_dir}, Prefix: {relabel_prefix}, Step: {step}, Seed: {seed:03d}"
        )

    if not os.path.exists(step_file):
        # Fallback: try without relabel prefix
        if relabel_percentage is not None:
            step_file_no_relabel = os.path.join(
                records_dir, f"step_{step}_{seed:03d}.pt"
            )
            if os.path.exists(step_file_no_relabel):
                if logger:
                    logger.debug(
                        f"Using non-relabel step file as fallback: {step_file_no_relabel}"
                    )
                step_file = step_file_no_relabel
            else:
                if logger:
                    logger.debug(
                        f"Fallback step file also not found: {step_file_no_relabel}"
                    )
                raise FileNotFoundError(
                    f"Could not find step file for step {step}, seed {seed} (tried w/ and w/o relabel prefix at {records_dir})"
                )
        else:
            raise FileNotFoundError(f"Could not find step file: {step_file}")

    return torch.load(step_file, weights_only=False)


def load_model_state_from_fallback(
    fn_fallback: str, target_state: str, device: str, logger: logging.Logger
) -> Dict:
    """Loads initial ('init') or final ('final') model state from the fallback .dat file."""
    logger.warning(
        f"Attempting to load {target_state} model state from fallback file: {fn_fallback}"
    )
    try:
        # weights_only=False is needed if the file contains NetList objects etc.
        res = torch.load(fn_fallback, map_location=device, weights_only=False)

        # Check 1: Is 'models' a list of models (old format)?
        # Need to carefully check if NetworkModule.NetList exists and is the correct type
        models_attr = res.get("models")
        is_netlist_like = hasattr(models_attr, "models") and isinstance(
            getattr(models_attr, "models", None), list
        )

        if is_netlist_like:
            models_list = models_attr.models
            if models_list:
                if target_state == "init":
                    state_dict = models_list[0].state_dict()
                    logger.info(
                        f"Loaded initial model state from fallback {fn_fallback} (models list index 0)"
                    )
                    return state_dict
                elif target_state == "final":
                    state_dict = models_list[-1].state_dict()
                    logger.info(
                        f"Loaded final model state from fallback {fn_fallback} (models list index -1)"
                    )
                    return state_dict
                else:
                    logger.error(
                        f"Unknown target_state '{target_state}' for fallback loading from models list."
                    )
                    raise ValueError("Invalid target_state for fallback model loading")
            else:
                logger.error(
                    f"Fallback file {fn_fallback} has 'models' but the list is empty."
                )
                raise ValueError("Empty models list in fallback file")

        # Check 2: Specific keys like 'init_model_state', 'final_model_state'
        elif target_state == "init" and "init_model_state" in res:
            state_dict = res["init_model_state"]
            logger.info(
                f"Loaded initial model state from fallback {fn_fallback} ('init_model_state')"
            )
            return state_dict
        elif target_state == "final" and "final_model_state" in res:
            state_dict = res["final_model_state"]
            logger.info(
                f"Loaded final model state from fallback {fn_fallback} ('final_model_state')"
            )
            return state_dict
        # Check 3: Generic keys like 'model_state' (could be init or final depending on context)
        elif (
            target_state == "init" and "model_state" in res
        ):  # Often used for initial state
            state_dict = res["model_state"]
            logger.info(
                f"Loaded initial model state from fallback {fn_fallback} ('model_state')"
            )
            return state_dict

        else:
            logger.error(
                f"Could not find suitable {target_state} model state structure in fallback file {fn_fallback}. Checked 'models' list, '{target_state}_model_state', 'model_state'."
            )
            raise FileNotFoundError(
                f"No suitable {target_state} model state found in fallback file."
            )
    except FileNotFoundError:
        logger.error(f"Fallback model file itself not found: {fn_fallback}")
        raise
    except Exception as e:
        logger.error(
            f"Error loading {target_state} model state from fallback {fn_fallback}: {e}"
        )
        raise


def load_initial_model(
    dn: str,
    seed: int,
    fn_fallback: str,
    device: str,
    logger: logging.Logger,
    relabel_percentage: float = None,
) -> Dict:
    """Loads the initial model state, trying records first, then fallback."""
    records_dir = os.path.join(dn, "records")
    relabel_prefix = (
        f"relabel_{int(relabel_percentage):03d}_pct_"
        if relabel_percentage is not None
        else ""
    )
    init_file_name = f"{relabel_prefix}init_{seed:03d}.pt"
    init_file_base = f"init_{seed:03d}.pt"  # Non-relabel name

    init_paths_to_try = [
        os.path.join(records_dir, init_file_name),
        os.path.join(records_dir, init_file_base),  # Try non-relabel name as fallback
    ]

    loaded_from_record = False
    for init_file in init_paths_to_try:
        if not os.path.exists(init_file):
            continue
        try:
            init_data = torch.load(init_file, map_location=device)
            state_dict = None
            if isinstance(init_data, dict) and "model_state" in init_data:
                state_dict = init_data["model_state"]
            elif isinstance(
                init_data, dict
            ):  # Allow loading if state_dict is the whole dict
                # Check if it looks like a state_dict (contains parameter keys) - basic check
                if any(
                    k.endswith(".weight") or k.endswith(".bias")
                    for k in init_data.keys()
                ):
                    state_dict = init_data
                else:
                    logger.warning(
                        f"Initial model file {init_file} is a dictionary but doesn't look like a state_dict. Keys: {list(init_data.keys())[:5]}..."
                    )
                    continue  # Try next file or fallback
            else:  # Assume it's the state_dict directly if not a dict (less common)
                logger.warning(
                    f"Initial model file {init_file} is not a dictionary. Assuming it's the state_dict directly."
                )
                # Add a check if it's actually a tensor dict
                if isinstance(init_data, dict) and all(
                    isinstance(v, torch.Tensor) for v in init_data.values()
                ):
                    state_dict = init_data
                else:
                    logger.error(
                        f"Initial model file {init_file} content is not a state_dict (type: {type(init_data)}). Cannot load."
                    )
                    continue  # Try next file or fallback

            if state_dict is not None:
                logger.info(f"Loaded initial model state from {init_file}")
                loaded_from_record = True
                return state_dict  # Success

        except Exception as e:
            logger.error(
                f"Error loading initial model state from {init_file}: {e}. Trying next or fallback."
            )
            # Continue to try other paths or fallback

    if not loaded_from_record:
        logger.warning(
            f"Initial model file not found in records (tried: {init_paths_to_try}). Trying fallback .dat."
        )
        return load_model_state_from_fallback(fn_fallback, "init", device, logger)


def load_final_model(
    dn: str,
    seed: int,
    global_info: Dict[str, Any],
    fn_fallback: str,
    device: str,
    logger: logging.Logger,
    relabel_percentage: float = None,
) -> Dict:
    """Loads the final model state, trying step/epoch files first, then fallback."""
    num_epochs = global_info["num_epoch"]
    batch_size_global = global_info["batch_size"]
    sample_num = global_info["n_tr"]
    steps_per_epoch = int(np.ceil(sample_num / batch_size_global))
    total_steps = num_epochs * steps_per_epoch
    last_step_index = total_steps  # The state *after* the last step
    last_epoch_index = num_epochs - 1  # State saved *after* the last epoch completed

    # 1. Try loading state after the absolute last step
    try:
        last_step_data = load_step_data(
            dn, last_step_index, seed, relabel_percentage, logger
        )
        if isinstance(last_step_data, dict) and "model_state" in last_step_data:
            state_dict = last_step_data["model_state"]
            logger.info(
                f"Successfully loaded final model state from step {last_step_index}"
            )
            return state_dict
        else:
            logger.warning(
                f"Loaded step file for step {last_step_index} does not contain 'model_state'."
            )
    except FileNotFoundError:
        logger.warning(f"Final step file ({last_step_index}) not found.")
    except Exception as e:
        logger.error(
            f"Error loading final model state from step file {last_step_index}: {e}."
        )

    # 2. Try loading from the last epoch file
    if last_epoch_index >= 0:  # Only if there was at least one epoch
        try:
            last_epoch_data = load_epoch_data(
                dn, last_epoch_index, seed, relabel_percentage, logger
            )
            if isinstance(last_epoch_data, dict) and "model_state" in last_epoch_data:
                state_dict = last_epoch_data["model_state"]
                logger.info(
                    f"Successfully loaded final model state from epoch {last_epoch_index}"
                )
                return state_dict
            else:
                logger.warning(
                    f"Loaded epoch file for epoch {last_epoch_index} does not contain 'model_state'."
                )
        except FileNotFoundError:
            logger.warning(f"Last epoch file ({last_epoch_index}) not found.")
        except Exception as e:
            logger.error(
                f"Error loading final model state from epoch file {last_epoch_index}: {e}."
            )

    # 3. Fall back to loading from the main results file
    logger.warning(
        "Could not load final model from step or epoch records. Trying fallback .dat file."
    )
    return load_model_state_from_fallback(fn_fallback, "final", device, logger)


def save_results(
    infl_data: Union[np.ndarray, List[np.ndarray]],
    dn: str,
    seed: int,
    infl_type: str,
    logger: logging.Logger,
    relabel_percentage: float = None,
):
    """Saves influence results to .dat and .csv files."""
    # Get output paths using the specific infl_type
    # Need key and model_type to generate base_dir if dn is not absolute/complete
    # Assuming dn is the correct base directory passed from the calculator instance
    _, _, gn_dat = get_file_paths(
        None, None, seed, infl_type, save_dir=dn, relabel_percentage=relabel_percentage
    )  # Use dn as save_dir

    relabel_prefix_csv = (
        f"relabel_{int(relabel_percentage):03d}_pct_"
        if relabel_percentage is not None
        else ""
    )
    csv_fn = os.path.join(dn, f"infl_{infl_type}_{relabel_prefix_csv}{seed:03d}.csv")

    os.makedirs(os.path.dirname(gn_dat), exist_ok=True)  # Ensure dir exists
    os.makedirs(os.path.dirname(csv_fn), exist_ok=True)

    # Save to .dat
    try:
        data_to_save = infl_data
        # Convert numpy arrays to tensors before saving if needed
        if isinstance(infl_data, np.ndarray):
            data_to_save = torch.from_numpy(infl_data)
        elif (
            isinstance(infl_data, list)
            and infl_data
            and isinstance(infl_data[0], np.ndarray)
        ):
            data_to_save = [torch.from_numpy(item) for item in infl_data]
        # else: Assume already tensor or list of tensors (or other saveable type)

        torch.save(data_to_save, gn_dat)
        logger.info(f"Influence results saved to .dat: {gn_dat}")
    except Exception as e:
        logger.error(f"Failed to save influence results to {gn_dat}: {e}")

    # Save to .csv
    try:
        df = None
        if isinstance(infl_data, list):  # List of arrays (e.g., per epoch/segment)
            num_items = len(infl_data)
            if num_items > 0 and isinstance(infl_data[0], (np.ndarray, torch.Tensor)):
                # Convert to numpy if tensors
                if isinstance(infl_data[0], torch.Tensor):
                    infl_data_np = [item.cpu().numpy() for item in infl_data]
                else:
                    infl_data_np = infl_data

                num_samples = len(infl_data_np[0])
                data_dict = {"sample_idx": np.arange(num_samples)}
                # Assuming list index corresponds to epoch or segment
                for i, infl_array in enumerate(infl_data_np):
                    # Check if segment is 1D or multi-dimensional
                    if infl_array.ndim == 1:
                        data_dict[f"influence_segment_{i}"] = infl_array
                    else:
                        # Handle multi-dimensional arrays if necessary (e.g., save flattened or skip)
                        logger.warning(
                            f"Segment {i} has shape {infl_array.shape}, saving flattened version to CSV."
                        )
                        data_dict[f"influence_segment_{i}_flat"] = (
                            infl_array.flatten()
                        )  # Example: flatten

                try:
                    df = pd.DataFrame(data_dict)
                except ValueError as ve:
                    logger.error(
                        f"Error creating DataFrame for list data (potential length mismatch?): {ve}"
                    )
                    # Fallback: Save segments individually or handle differently
                    df = pd.DataFrame({"error": [f"Could not create DataFrame: {ve}"]})

            else:
                df = pd.DataFrame({"sample_idx": []})  # Empty case
                logger.warning(
                    "Influence data list is empty or contains non-array items, saving empty/minimal CSV."
                )
        elif isinstance(infl_data, (np.ndarray, torch.Tensor)):  # Single array
            if isinstance(infl_data, torch.Tensor):
                infl_data_np = infl_data.cpu().numpy()
            else:
                infl_data_np = infl_data

            if infl_data_np.ndim == 1:  # Standard case
                num_samples = len(infl_data_np)
                df = pd.DataFrame(
                    {"sample_idx": np.arange(num_samples), "influence": infl_data_np}
                )
                # Add ranking info for single array results
                df["influence_rank"] = (
                    df["influence"].rank(ascending=False, method="first").astype(int)
                )
                df["influence_percentile"] = df["influence"].rank(pct=True)
            else:
                logger.warning(
                    f"Influence data is a multi-dimensional array (shape {infl_data_np.shape}). Saving flattened version to CSV."
                )
                num_samples = infl_data_np.shape[
                    0
                ]  # Assuming first dim is sample index
                df = pd.DataFrame(
                    {
                        "sample_idx": np.arange(num_samples),
                        "influence_flat": [
                            row.flatten().tolist() for row in infl_data_np
                        ],  # Example: save as list of lists/arrays
                    }
                )
                # Ranking might not make sense here or needs adaptation

        if df is not None:
            df.to_csv(csv_fn, index=False)
            logger.info(f"Influence results saved to CSV: {csv_fn}")
        else:
            logger.warning(
                f"Could not determine appropriate format to save influence data (type: {type(infl_data)}) to CSV."
            )

    except Exception as e:
        logger.error(f"Failed to save influence results to {csv_fn}: {e}")


# --- Hessian/HVP Helpers ---
def compute_adaptive_lambda(
    hessian_eigenvalues=None, u_norm=None, hu_norm=None, base_lambda=0.1, logger=None
):
    lambda_reg = base_lambda
    if hessian_eigenvalues is not None:
        max_abs_eig = max(
            abs(hessian_eigenvalues.max()), abs(hessian_eigenvalues.min())
        )
        lambda_reg = max(base_lambda, max_abs_eig * 0.01)  # 1% of max abs eigenvalue
        if logger:
            logger.debug(
                f"Adaptive lambda based on eigenvalues: {lambda_reg:.4e} (max_abs_eig: {max_abs_eig:.4e})"
            )
    elif (
        u_norm is not None and hu_norm is not None and u_norm > 1e-9
    ):  # Avoid division by zero
        ratio = hu_norm / u_norm
        # Heuristic adjustment: Dampen large increases, allow moderate ones
        if ratio > 10:  # If Hu is significantly larger than u
            # Scale lambda proportionally to the ratio, but maybe capped or scaled down
            scale_factor = min(
                max(1.0, ratio / 10.0), 100.0
            )  # Example: Scale up to 100x base
            lambda_reg = base_lambda * scale_factor
            if logger:
                logger.debug(
                    f"Adaptive lambda based on norm ratio: {lambda_reg:.4e} (ratio: {ratio:.2f}, scale: {scale_factor:.2f})"
                )
        else:
            if logger:
                logger.debug(
                    f"Adaptive lambda using base value: {lambda_reg:.4e} (ratio: {ratio:.2f} <= 10)"
                )
    else:
        if logger:
            logger.debug(
                f"Adaptive lambda using base value: {lambda_reg:.4e} (norms unavailable or u_norm near zero)"
            )
    return lambda_reg


def compute_hvp_with_finite_diff(model, x, y, u, loss_fn, alpha=0.0, epsilon=1e-5):
    """Computes Hessian-vector product using finite differences. Includes L2 reg."""
    device = next(model.parameters()).device
    u_device = [uu.to(device) for uu in u]  # Ensure u is on correct device

    # Store original parameters
    original_params = [p.clone().detach() for p in model.parameters()]
    params_list = list(model.parameters())  # Get list once

    # Compute loss and gradient at p + eps*u
    with torch.no_grad():
        for p, uu in zip(params_list, u_device):
            p.add_(epsilon * uu)
    model.zero_grad()
    loss_plus = loss_fn(model(x), y)
    if alpha > 0:
        l2_reg_plus = 0.0
        for p in params_list:
            l2_reg_plus += 0.5 * alpha * torch.sum(p * p)
        loss_plus += l2_reg_plus
    # grad_plus = torch.autograd.grad(loss_plus, params_list, allow_unused=True) # Causes issues with shared params?
    loss_plus.backward()
    grad_plus = [
        (p.grad.data.clone() if p.grad is not None else torch.zeros_like(p))
        for p in params_list
    ]

    # Reset parameters
    with torch.no_grad():
        for p, p_orig in zip(params_list, original_params):
            p.copy_(p_orig)

    # Compute loss and gradient at p - eps*u
    with torch.no_grad():
        for p, uu in zip(params_list, u_device):
            p.sub_(epsilon * uu)
    model.zero_grad()
    loss_minus = loss_fn(model(x), y)
    if alpha > 0:
        l2_reg_minus = 0.0
        for p in params_list:
            l2_reg_minus += 0.5 * alpha * torch.sum(p * p)
        loss_minus += l2_reg_minus
    # grad_minus = torch.autograd.grad(loss_minus, params_list, allow_unused=True) # Causes issues with shared params?
    loss_minus.backward()
    grad_minus = [
        (p.grad.data.clone() if p.grad is not None else torch.zeros_like(p))
        for p in params_list
    ]

    # Reset parameters (important!)
    with torch.no_grad():
        for p, p_orig in zip(params_list, original_params):
            p.copy_(p_orig)

    # Compute HVP approximation: (grad(p+eps*u) - grad(p-eps*u)) / (2*eps)
    hvp = [
        (gp.detach() - gm.detach()) / (2 * epsilon)
        for gp, gm in zip(grad_plus, grad_minus)
    ]

    # Clean grads on model just in case
    model.zero_grad()

    return hvp


# --- Influence Difference Helper (Used by _first/_middle/_last variants) ---
def infl_diff_helper(
    key: str,
    model_type: str,
    seed: int,
    gpu: int,
    save_dir: str,
    epoch_index: int,
    target_infl_type: str,
    source_infl_prefix: str,
    logger: logging.Logger,
    relabel_percentage: float = None,
):
    """Calculates influence difference between epoch_index and epoch_index+1."""
    logger.info(
        f"Starting infl_{target_infl_type} computation for {key}, {model_type}, seed {seed} at epoch {epoch_index}"
    )
    # Use save_dir directly if provided, otherwise construct standard path
    dn, fn_fallback, _ = get_file_paths(
        key, model_type, seed, None, save_dir, relabel_percentage
    )

    # Path to the *full* results saved by the source method (e.g., lie_full or segment_true_full)
    _, _, full_results_path = get_file_paths(
        key, model_type, seed, f"{source_infl_prefix}_full", dn, relabel_percentage
    )  # Use dn as save_dir here

    try:
        # Load onto CPU first to avoid GPU memory issues if large
        full_results = torch.load(full_results_path, map_location="cpu")
    except FileNotFoundError:
        logger.error(f"Required source results file not found: {full_results_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading source results file {full_results_path}: {e}")
        raise

    if not isinstance(full_results, list) or len(full_results) < 2:
        logger.error(
            f"Source results file {full_results_path} is not a list or has < 2 entries (found {len(full_results)}). Cannot compute difference."
        )
        raise ValueError(f"Invalid format or insufficient data in {full_results_path}.")

    # Convert elements to numpy if they are tensors
    try:
        if isinstance(full_results[0], torch.Tensor):
            full_results_np = [res.numpy() for res in full_results]
        elif isinstance(full_results[0], np.ndarray):
            full_results_np = full_results  # Already numpy
        else:
            raise TypeError(
                f"Elements in full_results are not Tensors or ndarrays (type: {type(full_results[0])})"
            )
    except Exception as e:
        logger.error(f"Error converting loaded results to NumPy arrays: {e}")
        raise

    num_results = len(full_results_np)
    # Allow epoch_index == num_results - 2 (calculates diff between last and second-to-last)
    if epoch_index < 0 or epoch_index > num_results - 2:
        logger.error(
            f"Epoch index {epoch_index} is out of range for difference calculation with {num_results} states (need index 0 to {num_results-2})."
        )
        raise ValueError(f"Epoch index {epoch_index} out of range for diff calc.")

    # Difference is state[epoch+1] - state[epoch]
    infl_diff = full_results_np[epoch_index + 1] - full_results_np[epoch_index]

    # Save the difference using the standard save function with the target_infl_type
    save_results(infl_diff, dn, seed, target_infl_type, logger, relabel_percentage)

    logger.info(
        f"Finished infl_{target_infl_type} computation for {key}, {model_type}, seed {seed}"
    )
    # No need to return infl_diff here, as it's saved.


# --- Base Class for Influence Calculators ---
class InfluenceCalculator(ABC):
    """Abstract Base Class for all influence function calculators."""

    def __init__(
        self,
        key: str,
        model_type: str,
        seed: int,
        gpu: int,
        save_dir: str,
        relabel_percentage: float,
        use_tensorboard: bool,
        **kwargs,
    ):
        self.key = key
        self.model_type = model_type
        self.seed = seed
        self.gpu = gpu
        self.save_dir_override = save_dir  # User-provided override
        self.relabel_percentage = relabel_percentage
        self.use_tensorboard = use_tensorboard
        self.kwargs = kwargs  # Store extra args like 'length'

        self.infl_type = self._get_infl_type()  # Get type from concrete class

        # Setup logger specific to this calculator instance
        self.logger = logging.getLogger(
            f"infl_{self.infl_type}_{self.key}_{self.model_type}_{self.seed}"
        )
        self.logger.info(f"[{self.__class__.__name__}] Initializing calculator...")

        self.device = get_device(self.gpu)

        # Common setup: Paths, global info, data, common params
        self._setup_common()

    @abstractmethod
    def _get_infl_type(self) -> str:
        """Return the string identifier for this influence type (e.g., 'sgd', 'true')."""
        pass

    def _setup_common(self):
        """Performs common setup steps: paths, global_info, data loading."""
        self.logger.info(f"Performing common setup...")
        self.dn, self.fn_fallback, self.gn_dat = get_file_paths(
            self.key,
            self.model_type,
            self.seed,
            self.infl_type,
            self.save_dir_override,
            self.relabel_percentage,
        )
        self.logger.info(f"Results directory: {self.dn}")
        self.logger.info(f"Fallback .dat path: {self.fn_fallback}")
        self.logger.info(f"Output .dat path: {self.gn_dat}")

        # Load global info (can still use helper)
        self.global_info = load_global_info(
            self.dn,
            self.seed,
            self.fn_fallback,
            self.device,
            self.logger,
            self.relabel_percentage,
        )
        self.logger.debug(f"Global Info: {self.global_info}")

        # Load data (load necessary splits - base loads train/val)
        self.x_tr, self.y_tr, self.x_val, self.y_val = load_data(
            self.key,
            self.global_info,
            self.seed,
            self.device,
            self.logger,
            self.relabel_percentage,
            self.dn,
        )
        self.logger.info(
            f"Data loaded: x_tr={self.x_tr.shape}, y_tr={self.y_tr.shape}, x_val={self.x_val.shape}, y_val={self.y_val.shape}"
        )

        # Common parameters from global_info
        self.input_dim = get_input_dim(
            self.x_tr, self.model_type
        )  # Use x_tr for consistent dim
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.alpha = self.global_info.get("alpha", 0.0)  # Default alpha to 0 if missing
        if "alpha" not in self.global_info:
            self.logger.warning("'alpha' not found in global_info, defaulting to 0.0.")
        self.n_tr = self.global_info["n_tr"]
        self.num_epochs = self.global_info["num_epoch"]
        self.batch_size_global = self.global_info["batch_size"]
        self.steps_per_epoch = int(np.ceil(self.n_tr / self.batch_size_global))
        self.total_steps = self.num_epochs * self.steps_per_epoch

        # Common TensorBoard setup
        self.tb_writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_log_dir = os.path.join(
                    self.dn, f"tensorboard_{self.infl_type}_{self.seed}"
                )
                self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
                self.logger.info(f"TensorBoard logging enabled: {tb_log_dir}")
            except ImportError:
                self.logger.warning(
                    "TensorBoard import failed, continuing without TensorBoard."
                )
        self.logger.info(f"Common setup complete.")

    @abstractmethod
    def calculate(self) -> Union[np.ndarray, List[np.ndarray]]:
        """The core influence calculation logic implemented by subclasses."""
        pass

    def _save(self, infl_data):
        """Helper to save results using the unified save_results function."""
        save_results(
            infl_data,
            self.dn,
            self.seed,
            self.infl_type,
            self.logger,
            self.relabel_percentage,
        )
        if self.tb_writer:
            self.tb_writer.close()  # Close writer after saving/calculation finishes

    def run(self):
        """Runs the calculation and saves the results."""
        self.logger.info(f"[{self.__class__.__name__}] Starting calculation...")
        try:
            result = self.calculate()
            self.logger.info(
                f"[{self.__class__.__name__}] Calculation finished. Saving results..."
            )
            # Use specific infl_type for saving if needed (e.g., lie_full)
            save_type = self.kwargs.get(
                "save_infl_type", self.infl_type
            )  # Allow override for special cases
            self._save(result)
            self.logger.info(
                f"[{self.__class__.__name__}] Results saved for type '{save_type}'."
            )
            # Clean up memory
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(
                f"[{self.__class__.__name__}] Calculation failed: {e}", exc_info=True
            )
            # Ensure TensorBoard writer is closed even on error
            if self.tb_writer:
                self.tb_writer.close()
            raise  # Re-raise the exception


# --- Factory Class ---
class InfluenceCalculatorFactory:
    """Factory for creating influence calculator instances."""

    _calculators: Dict[str, Type[InfluenceCalculator]] = {}  # Registry

    @classmethod
    def register(cls, infl_type: str):
        """Decorator to register influence calculator classes."""

        def decorator(calculator_class: Type[InfluenceCalculator]):
            if not issubclass(calculator_class, InfluenceCalculator):
                raise TypeError(
                    f"{calculator_class.__name__} must inherit from InfluenceCalculator"
                )
            if infl_type in cls._calculators:
                logging.warning(
                    f"Overwriting registration for influence type '{infl_type}'"
                )
            cls._calculators[infl_type] = calculator_class
            logging.getLogger(__name__).debug(
                f"Registered influence calculator: {infl_type} -> {calculator_class.__name__}"
            )
            return calculator_class

        return decorator

    @classmethod
    def create(cls, infl_type: str, **kwargs) -> InfluenceCalculator:
        """Creates an instance of the requested influence calculator."""
        calculator_class = cls._calculators.get(infl_type)
        if not calculator_class:
            raise ValueError(
                f"Unknown influence type: '{infl_type}'. Available types: {list(cls._calculators.keys())}"
            )

        # Pass all kwargs (key, model_type, seed, gpu, etc., plus specific ones like length)
        return calculator_class(
            infl_type=infl_type, **kwargs
        )  # Pass infl_type if needed by __init__? No, it's derived in base.


# --- Concrete Calculator Implementations ---


@InfluenceCalculatorFactory.register("sgd")
class SgdInfluenceCalculator(InfluenceCalculator):
    """Computes influence using the reverse SGD method."""

    def _get_infl_type(self) -> str:
        return "sgd"

    def calculate(self) -> np.ndarray:
        # --- Load initial model --- (Specific setup for SGD)
        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        model.load_state_dict(
            load_initial_model(
                self.dn,
                self.seed,
                self.fn_fallback,
                self.device,
                self.logger,
                self.relabel_percentage,
            )
        )
        model.eval()

        # --- Initial Gradient --- (Specific setup for SGD)
        u = compute_gradient(
            self.x_val, self.y_val, model, self.loss_fn
        )  # Grad w.r.t val loss at START
        u = [uu.to(self.device) for uu in u]
        try:
            u = [uu.to(torch.float64) for uu in u]
            self.logger.info("Using float64 precision for u vector.")
        except TypeError:
            self.logger.warning(
                "float64 not supported for u vector, falling back to float32."
            )
            u = [uu.to(torch.float32) for uu in u]

        infl = np.zeros(self.n_tr, dtype=np.float64)

        # --- Core Logic: Reverse SGD ---
        for t in range(self.total_steps, 0, -1):  # Process steps in reverse
            step_log_prefix = f"Step {t}/{self.total_steps}"
            try:
                # Load data for the state *before* this step t (which is saved as step_t.pt)
                step_data = load_step_data(
                    self.dn, t, self.seed, self.relabel_percentage, self.logger
                )  # Loads state *after* step t-1
                current_model_state = step_data["model_state"]
                idx, lr = step_data["idx"], step_data["lr"]  # Info for step t update

                # Temporary model instance for this step's state
                m_step = get_network(
                    self.model_type, self.input_dim, logger=self.logger
                ).to(self.device)
                m_step.load_state_dict(current_model_state)
                m_step.eval()

                if not isinstance(idx, (list, np.ndarray, torch.Tensor)):
                    idx = [idx]
                idx = torch.tensor(
                    idx, device=self.device
                )  # Ensure idx is tensor on correct device
                if len(idx) == 0:
                    self.logger.warning(
                        f"{step_log_prefix}: Batch index list 'idx' is empty. Skipping."
                    )
                    continue

                # Filter out invalid indices
                valid_idx_mask = (idx >= 0) & (idx < self.n_tr)
                if not valid_idx_mask.all():
                    self.logger.warning(
                        f"{step_log_prefix}: Found out-of-bounds indices ({idx[~valid_idx_mask].tolist()}). Using only valid indices."
                    )
                    idx = idx[valid_idx_mask]
                if len(idx) == 0:
                    self.logger.warning(
                        f"{step_log_prefix}: No valid indices left after filtering. Skipping."
                    )
                    continue

                batch_size = len(idx)
                x_batch, y_batch = self.x_tr[idx], self.y_tr[idx]

                # 1. Accumulate influence infl[i] using state m_step (before step t)
                # Optimize grad calculation: Compute per-sample grads for the batch once
                m_step.zero_grad()
                z_batch = m_step(x_batch)
                loss_indiv = self.loss_fn(
                    z_batch, y_batch
                )  # Example: Assumes loss can handle batch (adjust if per-sample needed)
                # Need per-sample gradients. This is tricky without specific libraries (like backpack).
                # Approximation: Calculate gradient of average loss, scale later? Less accurate.
                # Manual loop for per-sample grad is slow but accurate:
                param_grads_list = []
                for i_local in range(batch_size):
                    m_step.zero_grad()
                    z_i = m_step(x_batch[[i_local]])
                    loss_i = self.loss_fn(z_i, y_batch[[i_local]])
                    if self.alpha > 0:
                        l2_reg_i = 0.0
                        for p in m_step.parameters():
                            l2_reg_i += 0.5 * self.alpha * (p * p).sum()
                        loss_i += (
                            l2_reg_i / batch_size
                        )  # Scale L2 reg contribution approx? Or add full L2? Paper consistency needed. Let's add full L2 as in original code.
                        loss_i = self.loss_fn(
                            z_i, y_batch[[i_local]]
                        )  # Recompute base loss
                        if self.alpha > 0:
                            for p in m_step.parameters():
                                loss_i += 0.5 * self.alpha * (p * p).sum()

                    loss_i.backward()
                    grad_i = [
                        (
                            p.grad.data.clone().to(dtype=u[0].dtype)
                            if p.grad is not None
                            else torch.zeros_like(p, dtype=u[0].dtype)
                        )
                        for p in m_step.parameters()
                    ]
                    param_grads_list.append(grad_i)

                m_step.zero_grad()  # Clear grads after loop

                for i_local, sample_idx in enumerate(
                    idx.tolist()
                ):  # Use .tolist() for numpy indexing
                    grad_i = param_grads_list[i_local]
                    grad_sum = 0.0
                    for j, param_grad in enumerate(grad_i):
                        if j < len(u):  # Ensure index exists
                            grad_sum += torch.sum(
                                u[j].data * param_grad
                            ).item()  # Ensure same dtype
                        else:
                            self.logger.error(
                                f"Index mismatch: u has {len(u)} elements, gradient has {len(grad_i)} at parameter index {j}"
                            )
                    # Note: Original code normalized by len(idx) here.
                    # Check paper: Influence is often defined without 1/N, but depends on exact formula. Assuming original normalization.
                    infl[sample_idx] += lr * grad_sum / batch_size

                # 2. Update vector u using HVP at state m_step
                u_prev = [uu.clone() for uu in u]
                # HVP uses batch data
                hvp = compute_hvp_with_finite_diff(
                    m_step, x_batch, y_batch, u, self.loss_fn, alpha=self.alpha
                )
                lambda_reg = compute_adaptive_lambda(
                    u_norm=sum_norm(u),
                    hu_norm=sum_norm(hvp),
                    base_lambda=0.1,
                    logger=self.logger,
                )
                hvp_reg = [
                    hv.to(dtype=u[0].dtype) + lambda_reg * uu for hv, uu in zip(hvp, u)
                ]  # Ensure same dtype

                # Update u = u - lr * HVP_regularized
                for j in range(len(u)):
                    new_u_val = u[j] - lr * hvp_reg[j]
                    if torch.isnan(new_u_val).any() or torch.isinf(new_u_val).any():
                        self.logger.warning(
                            f"{step_log_prefix}: NaN/Inf in u[{j}] update, resetting to previous value. Norm: {u_prev[j].norm().item():.4e}"
                        )
                        u[j] = u_prev[j]
                    else:
                        # Optional: Stability check/clipping
                        old_norm = u[j].norm().item()
                        new_norm = new_u_val.norm().item()
                        if (
                            old_norm > 1e-9 and new_norm / old_norm > 100.0
                        ):  # Clip if norm increases > 100x (adjust threshold)
                            scale = 100.0 * old_norm / new_norm
                            u[j] = u[j] - scale * lr * hvp_reg[j]
                            self.logger.warning(
                                f"{step_log_prefix}: Clipped u[{j}] update, norm ratio {new_norm/old_norm:.2f}"
                            )
                        else:
                            u[j] = new_u_val

                del m_step  # Free memory
                if t % 50 == 0:
                    self.logger.info(
                        f"{step_log_prefix} processed. Current u norm: {sum_norm(u):.4e}, lambda_reg: {lambda_reg:.4e}"
                    )

                # TensorBoard Logging within loop
                if self.tb_writer is not None:  # Log every N steps
                    if (
                        self.total_steps - t
                    ) % 50 == 0:  # Log based on forward step count
                        global_step_for_tb = self.total_steps - t
                        self.tb_writer.add_scalar(
                            f"{self.infl_type}/u_norm", sum_norm(u), global_step_for_tb
                        )
                        self.tb_writer.add_scalar(
                            f"{self.infl_type}/lambda_reg",
                            lambda_reg,
                            global_step_for_tb,
                        )
                        # Add more histograms or details if needed
                        # for i_u, u_param in enumerate(u):
                        #     self.tb_writer.add_histogram(f'{self.infl_type}/u_{i_u}/values', u_param, global_step_for_tb)

            except FileNotFoundError:
                self.logger.warning(
                    f"{step_log_prefix}: Step file not found. Trying epoch fallback (less precise)."
                )
                # --- Epoch Fallback Logic ---
                try:
                    epoch_idx = (
                        t - 1
                    ) // self.steps_per_epoch  # Epoch containing state *before* step t
                    epoch_data = load_epoch_data(
                        self.dn,
                        epoch_idx,
                        self.seed,
                        self.relabel_percentage,
                        self.logger,
                    )
                    m_epoch = get_network(
                        self.model_type, self.input_dim, logger=self.logger
                    ).to(self.device)
                    m_epoch.load_state_dict(
                        epoch_data["model_state"]
                    )  # State at end of epoch_idx
                    m_epoch.eval()

                    # Try to get step info from epoch data
                    step_in_epoch_idx = (t - 1) % self.steps_per_epoch
                    if "step_info" in epoch_data and step_in_epoch_idx < len(
                        epoch_data["step_info"]
                    ):
                        step_info = epoch_data["step_info"][step_in_epoch_idx]
                        idx_epoch, lr_epoch = step_info["idx"], step_info["lr"]

                        if not isinstance(idx_epoch, (list, np.ndarray, torch.Tensor)):
                            idx_epoch = [idx_epoch]
                        idx_epoch = torch.tensor(idx_epoch, device=self.device)
                        if len(idx_epoch) == 0:
                            self.logger.warning(
                                f"{step_log_prefix}: Epoch fallback idx is empty. Skipping."
                            )
                            continue

                        valid_idx_mask = (idx_epoch >= 0) & (idx_epoch < self.n_tr)
                        idx_epoch = idx_epoch[valid_idx_mask]
                        if len(idx_epoch) == 0:
                            self.logger.warning(
                                f"{step_log_prefix}: No valid indices left in epoch fallback. Skipping."
                            )
                            continue

                        batch_size_epoch = len(idx_epoch)
                        x_batch_epoch, y_batch_epoch = (
                            self.x_tr[idx_epoch],
                            self.y_tr[idx_epoch],
                        )

                        # Recalculate steps 1 & 2 using the potentially stale epoch model state 'm_epoch'
                        # Step 1: Accumulate influence
                        param_grads_list_epoch = []
                        for i_local in range(batch_size_epoch):
                            m_epoch.zero_grad()
                            z_i = m_epoch(x_batch_epoch[[i_local]])
                            loss_i = self.loss_fn(z_i, y_batch_epoch[[i_local]])
                            if self.alpha > 0:
                                for p in m_epoch.parameters():
                                    loss_i += 0.5 * self.alpha * (p * p).sum()
                            loss_i.backward()
                            grad_i = [
                                (
                                    p.grad.data.clone().to(dtype=u[0].dtype)
                                    if p.grad is not None
                                    else torch.zeros_like(p, dtype=u[0].dtype)
                                )
                                for p in m_epoch.parameters()
                            ]
                            param_grads_list_epoch.append(grad_i)
                        m_epoch.zero_grad()

                        for i_local, sample_idx in enumerate(idx_epoch.tolist()):
                            grad_i = param_grads_list_epoch[i_local]
                            grad_sum = sum(
                                torch.sum(u[j].data * param_grad).item()
                                for j, param_grad in enumerate(grad_i)
                                if j < len(u)
                            )
                            infl[sample_idx] += lr_epoch * grad_sum / batch_size_epoch

                        # Step 2: Update u
                        u_prev = [uu.clone() for uu in u]
                        hvp_epoch = compute_hvp_with_finite_diff(
                            m_epoch,
                            x_batch_epoch,
                            y_batch_epoch,
                            u,
                            self.loss_fn,
                            alpha=self.alpha,
                        )
                        lambda_reg_epoch = compute_adaptive_lambda(
                            u_norm=sum_norm(u),
                            hu_norm=sum_norm(hvp_epoch),
                            base_lambda=0.1,
                            logger=self.logger,
                        )
                        hvp_reg_epoch = [
                            hv.to(dtype=u[0].dtype) + lambda_reg_epoch * uu
                            for hv, uu in zip(hvp_epoch, u)
                        ]

                        for j in range(len(u)):
                            new_u_val = u[j] - lr_epoch * hvp_reg_epoch[j]
                            if (
                                torch.isnan(new_u_val).any()
                                or torch.isinf(new_u_val).any()
                            ):
                                u[j] = u_prev[j]
                            else:
                                u[j] = new_u_val  # Simplified update for fallback

                        self.logger.info(
                            f"{step_log_prefix}: Processed using epoch {epoch_idx} fallback data."
                        )
                        del m_epoch

                    else:
                        self.logger.warning(
                            f"{step_log_prefix}: Step info index {step_in_epoch_idx} not found in epoch {epoch_idx} data or 'step_info' key missing. Cannot use epoch fallback. Skipping step."
                        )
                        continue
                except FileNotFoundError:
                    self.logger.error(
                        f"{step_log_prefix}: Could not find step file or epoch fallback file (epoch {epoch_idx}). Skipping step."
                    )
                    continue
                except Exception as e_inner:
                    self.logger.error(
                        f"{step_log_prefix}: Error during epoch fallback: {e_inner}",
                        exc_info=True,
                    )
                    continue
                # --- End Epoch Fallback ---

            except Exception as e_outer:
                self.logger.error(
                    f"{step_log_prefix}: Unexpected error processing step: {e_outer}",
                    exc_info=True,
                )
                continue  # Skip step on error

            # Optional: Clear GPU cache periodically
            if t % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # --- End Core Logic ---
        return infl


@InfluenceCalculatorFactory.register("nohess")
class NoHessInfluenceCalculator(InfluenceCalculator):
    """Computes influence using reverse SGD but ignoring the HVP term (u is fixed)."""

    def _get_infl_type(self) -> str:
        return "nohess"

    def calculate(self) -> np.ndarray:
        # --- Load FINAL model --- (Specific setup for NoHess)
        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        model.load_state_dict(
            load_final_model(
                self.dn,
                self.seed,
                self.global_info,
                self.fn_fallback,
                self.device,
                self.logger,
                self.relabel_percentage,
            )
        )
        model.eval()

        # --- Final Gradient --- (u is calculated once and fixed)
        u = compute_gradient(
            self.x_val, self.y_val, model, self.loss_fn
        )  # Grad w.r.t val loss at END
        u = [uu.to(self.device) for uu in u]
        try:
            u = [uu.to(torch.float64) for uu in u]
            self.logger.info("Using float64 precision for fixed u vector.")
        except TypeError:
            self.logger.warning(
                "float64 not supported for u vector, falling back to float32."
            )
            u = [uu.to(torch.float32) for uu in u]

        infl = np.zeros(self.n_tr, dtype=np.float64)

        # --- Core Logic: Reverse SGD without HVP/u update ---
        for t in range(self.total_steps, 0, -1):
            step_log_prefix = f"Step {t}/{self.total_steps}"
            try:
                step_data = load_step_data(
                    self.dn, t, self.seed, self.relabel_percentage, self.logger
                )
                current_model_state = step_data["model_state"]
                idx, lr = step_data["idx"], step_data["lr"]

                m_step = get_network(
                    self.model_type, self.input_dim, logger=self.logger
                ).to(self.device)
                m_step.load_state_dict(current_model_state)
                m_step.eval()

                if not isinstance(idx, (list, np.ndarray, torch.Tensor)):
                    idx = [idx]
                idx = torch.tensor(idx, device=self.device)
                if len(idx) == 0:
                    continue

                valid_idx_mask = (idx >= 0) & (idx < self.n_tr)
                idx = idx[valid_idx_mask]
                if len(idx) == 0:
                    continue

                batch_size = len(idx)
                x_batch, y_batch = self.x_tr[idx], self.y_tr[idx]

                # Accumulate influence (only step 1, u is fixed)
                # Use optimized per-sample grad loop from SGD calculator
                param_grads_list = []
                for i_local in range(batch_size):
                    m_step.zero_grad()
                    z_i = m_step(x_batch[[i_local]])
                    loss_i = self.loss_fn(z_i, y_batch[[i_local]])
                    if self.alpha > 0:
                        for p in m_step.parameters():
                            loss_i += 0.5 * self.alpha * (p * p).sum()
                    loss_i.backward()
                    grad_i = [
                        (
                            p.grad.data.clone().to(dtype=u[0].dtype)
                            if p.grad is not None
                            else torch.zeros_like(p, dtype=u[0].dtype)
                        )
                        for p in m_step.parameters()
                    ]
                    param_grads_list.append(grad_i)
                m_step.zero_grad()

                for i_local, sample_idx in enumerate(idx.tolist()):
                    grad_i = param_grads_list[i_local]
                    grad_sum = sum(
                        torch.sum(u[j].data * param_grad).item()
                        for j, param_grad in enumerate(grad_i)
                        if j < len(u)
                    )
                    infl[sample_idx] += lr * grad_sum / batch_size  # Use fixed u

                del m_step
                if t % 50 == 0:
                    self.logger.info(f"{step_log_prefix} processed.")

            except FileNotFoundError:
                self.logger.warning(
                    f"{step_log_prefix}: Step file not found. Skipping step (epoch fallback not standard for nohess)."
                )
                continue
            except Exception as e:
                self.logger.error(
                    f"{step_log_prefix}: Error processing step: {e}", exc_info=True
                )
                continue

            if t % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # --- End Core Logic ---
        return infl


@InfluenceCalculatorFactory.register("tim_last")
class TimLastInfluenceCalculator(InfluenceCalculator):
    """Computes influence by reversing SGD for the last 'length' epochs."""

    def _get_infl_type(self) -> str:
        return "tim_last"

    def calculate(self) -> np.ndarray:
        length = self.kwargs.get("length", 3)  # Get length from kwargs, default 3
        self.logger.info(f"Calculating TIM influence for last {length} epochs.")

        # --- Load FINAL model --- (Specific setup for TIM)
        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        model.load_state_dict(
            load_final_model(
                self.dn,
                self.seed,
                self.global_info,
                self.fn_fallback,
                self.device,
                self.logger,
                self.relabel_percentage,
            )
        )
        model.eval()

        # --- Final Gradient --- (u is calculated at final state)
        u = compute_gradient(self.x_val, self.y_val, model, self.loss_fn)
        u = [uu.to(self.device) for uu in u]
        try:
            u = [uu.to(torch.float64) for uu in u]
            self.logger.info("Using float64 precision for u vector.")
        except TypeError:
            self.logger.warning(
                "float64 not supported for u vector, falling back to float32"
            )
            u = [uu.to(torch.float32) for uu in u]

        infl = np.zeros(self.n_tr, dtype=np.float64)
        start_step_incl = self.total_steps  # Start from state *after* last step
        end_step_excl = max(
            0, self.total_steps - length * self.steps_per_epoch
        )  # Go back 'length' epochs

        self.logger.info(
            f"Reversing SGD from step {start_step_incl} back to step {end_step_excl + 1} (exclusive end)"
        )

        # --- Core Logic: Reverse SGD for last 'length' epochs ---
        # (Essentially the same loop as SGD, but over a shorter range and starting u from final state)
        for t in range(start_step_incl, end_step_excl, -1):
            step_log_prefix = f"Step {t}/{self.total_steps}"
            try:
                # Load data for state *before* step t
                step_data = load_step_data(
                    self.dn, t, self.seed, self.relabel_percentage, self.logger
                )
                current_model_state = step_data["model_state"]
                idx, lr = step_data["idx"], step_data["lr"]

                m_step = get_network(
                    self.model_type, self.input_dim, logger=self.logger
                ).to(self.device)
                m_step.load_state_dict(current_model_state)
                m_step.eval()

                if not isinstance(idx, (list, np.ndarray, torch.Tensor)):
                    idx = [idx]
                idx = torch.tensor(idx, device=self.device)
                if len(idx) == 0:
                    continue

                valid_idx_mask = (idx >= 0) & (idx < self.n_tr)
                idx = idx[valid_idx_mask]
                if len(idx) == 0:
                    continue

                batch_size = len(idx)
                x_batch, y_batch = self.x_tr[idx], self.y_tr[idx]

                # Step 1: Accumulate influence
                param_grads_list = []
                for i_local in range(batch_size):
                    m_step.zero_grad()
                    z_i = m_step(x_batch[[i_local]])
                    loss_i = self.loss_fn(z_i, y_batch[[i_local]])
                    if self.alpha > 0:
                        for p in m_step.parameters():
                            loss_i += 0.5 * self.alpha * (p * p).sum()
                    loss_i.backward()
                    grad_i = [
                        (
                            p.grad.data.clone().to(dtype=u[0].dtype)
                            if p.grad is not None
                            else torch.zeros_like(p, dtype=u[0].dtype)
                        )
                        for p in m_step.parameters()
                    ]
                    param_grads_list.append(grad_i)
                m_step.zero_grad()

                for i_local, sample_idx in enumerate(idx.tolist()):
                    grad_i = param_grads_list[i_local]
                    grad_sum = sum(
                        torch.sum(u[j].data * param_grad).item()
                        for j, param_grad in enumerate(grad_i)
                        if j < len(u)
                    )
                    infl[sample_idx] += lr * grad_sum / batch_size

                # Step 2: Update u
                u_prev = [uu.clone() for uu in u]
                hvp = compute_hvp_with_finite_diff(
                    m_step, x_batch, y_batch, u, self.loss_fn, alpha=self.alpha
                )
                lambda_reg = compute_adaptive_lambda(
                    u_norm=sum_norm(u),
                    hu_norm=sum_norm(hvp),
                    base_lambda=0.1,
                    logger=self.logger,
                )
                hvp_reg = [
                    hv.to(dtype=u[0].dtype) + lambda_reg * uu for hv, uu in zip(hvp, u)
                ]

                for j in range(len(u)):
                    new_u_val = u[j] - lr * hvp_reg[j]
                    if torch.isnan(new_u_val).any() or torch.isinf(new_u_val).any():
                        self.logger.warning(
                            f"{step_log_prefix}: NaN/Inf in u[{j}] update, resetting."
                        )
                        u[j] = u_prev[j]
                    else:
                        old_norm = u[j].norm().item()
                        new_norm = new_u_val.norm().item()
                        if old_norm > 1e-9 and new_norm / old_norm > 100.0:
                            scale = 100.0 * old_norm / new_norm
                            u[j] = u[j] - scale * lr * hvp_reg[j]
                            self.logger.warning(
                                f"{step_log_prefix}: Clipped u[{j}] update, norm ratio {new_norm/old_norm:.2f}"
                            )
                        else:
                            u[j] = new_u_val

                del m_step
                if t % 50 == 0:
                    self.logger.info(
                        f"{step_log_prefix} processed. Current u norm: {sum_norm(u):.4e}"
                    )

                # TensorBoard
                if self.tb_writer is not None:
                    if (
                        self.total_steps - t
                    ) % 50 == 0:  # Log based on forward step count relative to calculation range? Or absolute step?
                        global_step_for_tb = (
                            start_step_incl - t
                        )  # Steps processed in this calculation
                        self.tb_writer.add_scalar(
                            f"{self.infl_type}/u_norm", sum_norm(u), global_step_for_tb
                        )
                        self.tb_writer.add_scalar(
                            f"{self.infl_type}/lambda_reg",
                            lambda_reg,
                            global_step_for_tb,
                        )

            except FileNotFoundError:
                self.logger.warning(
                    f"{step_log_prefix}: Step file not found. Skipping step (epoch fallback not ideal for tim_last)."
                )
                continue
            except Exception as e:
                self.logger.error(
                    f"{step_log_prefix}: Error processing step: {e}", exc_info=True
                )
                continue

            if t % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # --- End Core Logic ---
        return infl


@InfluenceCalculatorFactory.register("true")
class TrueInfluenceCalculator(InfluenceCalculator):
    """Computes 'true' influence by comparing loss with counterfactual models."""

    def _get_infl_type(self) -> str:
        return "true"

    def calculate(self) -> np.ndarray:
        # --- Load Full Results Object (Needed for counterfactuals) ---
        self.logger.info(
            f"Loading full results object from fallback file: {self.fn_fallback}"
        )
        try:
            # Load onto CPU first to potentially save GPU memory
            res = torch.load(self.fn_fallback, map_location="cpu", weights_only=False)
            self.logger.info(f"Successfully loaded main results object.")
        except FileNotFoundError:
            self.logger.error(
                f"Required results file {self.fn_fallback} not found for 'true' influence."
            )
            raise
        except Exception as e:
            self.logger.error(f"Error loading results file {self.fn_fallback}: {e}")
            raise

        # --- Load Final Model State from res ---
        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        try:
            # Assume final model is last in 'models' list within res
            # Check if 'models' attribute and its 'models' list exist
            models_attr = res.get("models")
            is_netlist_like = hasattr(models_attr, "models") and isinstance(
                getattr(models_attr, "models", None), list
            )

            if is_netlist_like and models_attr.models:
                final_model_state = models_attr.models[-1].state_dict()
            elif "final_model_state" in res:  # Check for specific key
                final_model_state = res["final_model_state"]
            else:  # Add more fallbacks if necessary
                raise KeyError(
                    "Could not find final model state in 'res' object (checked res['models'].models[-1] and res['final_model_state'])"
                )

            model.load_state_dict(final_model_state)
            model.eval()
            self.logger.info("Loaded final model state from results object.")
        except (KeyError, AttributeError, IndexError) as e:
            self.logger.error(
                f"Could not extract final model state from {self.fn_fallback}: {e}"
            )
            raise ValueError(f"Cannot load final model from {self.fn_fallback}")

        # --- Core Logic: Counterfactual Comparison ---
        if "counterfactual" not in res or not isinstance(res["counterfactual"], list):
            self.logger.error(
                f"Counterfactual models not found or invalid in {self.fn_fallback}. Cannot compute 'true' influence."
            )
            raise ValueError("Missing or invalid counterfactual data.")

        counterfactuals = res["counterfactual"]
        num_counterfactuals = len(counterfactuals)
        if num_counterfactuals < self.n_tr:
            self.logger.warning(
                f"Found {num_counterfactuals} counterfactuals, but expected {self.n_tr}. Influence for missing indices will be 0."
            )

        infl = np.zeros(self.n_tr, dtype=np.float64)

        # Baseline loss with original final model
        with torch.no_grad():
            z_base = model(self.x_val)  # x_val should already be on device
            base_loss = self.loss_fn(
                z_base, self.y_val
            ).item()  # y_val should be on device
            self.logger.info(f"Base loss (final model): {base_loss:.6f}")

        # Temporary model for counterfactuals
        m_counterfactual = get_network(
            self.model_type, self.input_dim, logger=self.logger
        ).to(self.device)

        # Calculate influence by comparing loss with counterfactual models
        for i in range(self.n_tr):
            if i >= num_counterfactuals:
                # logger.warning(f"Missing counterfactual model for index {i}. Setting influence to 0.") # Logged once above
                infl[i] = 0.0
                continue

            counterfactual_res = counterfactuals[i]
            if counterfactual_res is None:
                self.logger.warning(
                    f"Counterfactual result for index {i} is None. Setting influence to 0."
                )
                infl[i] = 0.0
                continue

            try:
                # Assume counterfactual structure is res['counterfactual'][i].models[-1]
                cf_models_attr = getattr(counterfactual_res, "models", None)
                is_cf_netlist_like = hasattr(cf_models_attr, "models") and isinstance(
                    getattr(cf_models_attr, "models", None), list
                )

                if is_cf_netlist_like and cf_models_attr.models:
                    counterfactual_model_obj = cf_models_attr.models[-1]
                    # Move model to device *if it's not already there* (it was loaded to CPU)
                    counterfactual_model_obj.to(self.device)
                    m_counterfactual.load_state_dict(
                        counterfactual_model_obj.state_dict()
                    )  # Load state into reusable model
                elif (
                    isinstance(counterfactual_res, dict)
                    and "final_model_state" in counterfactual_res
                ):
                    # Alternative structure: dict with final state
                    m_counterfactual.load_state_dict(
                        counterfactual_res["final_model_state"]
                    )
                else:
                    # Maybe the counterfactual object *is* the model?
                    if hasattr(counterfactual_res, "state_dict") and callable(
                        counterfactual_res.state_dict
                    ):
                        counterfactual_res.to(self.device)  # Move to device
                        m_counterfactual.load_state_dict(
                            counterfactual_res.state_dict()
                        )
                    else:
                        raise AttributeError(
                            f"Cannot determine counterfactual model structure for index {i}. Object type: {type(counterfactual_res)}"
                        )

                m_counterfactual.eval()
                with torch.no_grad():
                    zi = m_counterfactual(self.x_val)
                    lossi = self.loss_fn(zi, self.y_val)
                infl[i] = (lossi - base_loss).item()

            except (AttributeError, IndexError, KeyError, TypeError) as e_inner:
                self.logger.warning(
                    f"Error processing counterfactual for index {i}: {e_inner}. Structure might be invalid. Setting influence to 0."
                )
                infl[i] = 0.0
            # except Exception as e_outer:
            #     self.logger.warning(f"Unexpected error processing counterfactual for index {i}: {e_outer}. Setting influence to 0.")
            #     infl[i] = 0.0

            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i+1}/{self.n_tr} counterfactuals.")
        # --- End Core Logic ---

        # Clean up large objects
        del res, counterfactuals, model, m_counterfactual
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return infl


# --- Calculators for difference methods ---
# These don't calculate directly but trigger the helper based on existing full results


class BaseDifferenceCalculator(InfluenceCalculator):
    """Base class for calculators deriving results from existing full calculations."""

    # These don't perform a full calculation, just trigger the diff helper.
    # The common setup might load unnecessary data (x_tr, y_tr). Override if needed.

    @abstractmethod
    def get_source_prefix(self) -> str:
        """Return the prefix of the full results file (e.g., 'lie', 'segment_true')."""
        pass

    @abstractmethod
    def get_epoch_index(self, full_results_len: int) -> int:
        """Return the epoch index to use for the difference calculation."""
        pass

    def _setup_common(self):
        # Override common setup - only need paths and logger for diff helper
        self.logger.info(f"Performing minimal setup for difference calculation...")
        self.dn, self.fn_fallback, _ = get_file_paths(
            self.key,
            self.model_type,
            self.seed,
            self.infl_type,
            self.save_dir_override,
            self.relabel_percentage,
        )
        self.logger.info(f"Results directory: {self.dn}")
        # No need to load global info, data, etc.

    def calculate(self) -> None:
        # The actual work is done by the helper function.
        # We need to determine the epoch index first, which might require loading the source file.
        source_prefix = self.get_source_prefix()
        _, _, full_results_path = get_file_paths(
            self.key,
            self.model_type,
            self.seed,
            f"{source_prefix}_full",
            self.dn,
            self.relabel_percentage,
        )

        try:
            # Load just enough to get the length
            full_results = torch.load(full_results_path, map_location="cpu")
            if not isinstance(full_results, list):
                raise ValueError(
                    f"Source results file {full_results_path} is not a list."
                )
            full_results_len = len(full_results)
            epoch_index = self.get_epoch_index(full_results_len)

            # Now call the helper
            infl_diff_helper(
                key=self.key,
                model_type=self.model_type,
                seed=self.seed,
                gpu=self.gpu,  # GPU not really used by helper, but pass for consistency
                save_dir=self.dn,  # Pass the derived directory
                epoch_index=epoch_index,
                target_infl_type=self.infl_type,  # The type we are calculating (e.g., 'tim_first')
                source_infl_prefix=source_prefix,  # The type of the source file (e.g., 'lie')
                logger=self.logger,
                relabel_percentage=self.relabel_percentage,
            )
            # Return None as the result is saved by the helper
            return None

        except FileNotFoundError:
            self.logger.error(
                f"Required source results file ({full_results_path}) not found for '{self.infl_type}'. Cannot calculate difference."
            )
            raise
        except ValueError as ve:
            self.logger.error(
                f"Error processing source file or calculating epoch index for '{self.infl_type}': {ve}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error during '{self.infl_type}' calculation: {e}"
            )
            raise


@InfluenceCalculatorFactory.register("tim_first")
class TimFirstInfluenceCalculator(BaseDifferenceCalculator):
    def _get_infl_type(self) -> str:
        return "tim_first"

    def get_source_prefix(self) -> str:
        return "lie"

    def get_epoch_index(self, full_results_len: int) -> int:
        return 0


@InfluenceCalculatorFactory.register("tim_middle")
class TimMiddleInfluenceCalculator(BaseDifferenceCalculator):
    def _get_infl_type(self) -> str:
        return "tim_middle"

    def get_source_prefix(self) -> str:
        return "lie"

    def get_epoch_index(self, full_results_len: int) -> int:
        if full_results_len < 2:
            raise ValueError("Need at least 2 states for difference.")
        # Calculate diff between state[mid] and state[mid-1]
        # Index required is mid-1
        middle_epoch_plus_1 = full_results_len // 2  # Index of state at middle point
        epoch_index = middle_epoch_plus_1 - 1
        if epoch_index < 0:
            raise ValueError("Not enough epochs for middle calculation")
        self.logger.info(
            f"Calculating middle difference between epoch {epoch_index+1} and {epoch_index}"
        )
        return epoch_index


@InfluenceCalculatorFactory.register("true_first")
class TrueFirstInfluenceCalculator(BaseDifferenceCalculator):
    def _get_infl_type(self) -> str:
        return "true_first"

    def get_source_prefix(self) -> str:
        return "segment_true"

    def get_epoch_index(self, full_results_len: int) -> int:
        return 0


@InfluenceCalculatorFactory.register("true_middle")
class TrueMiddleInfluenceCalculator(BaseDifferenceCalculator):
    def _get_infl_type(self) -> str:
        return "true_middle"

    def get_source_prefix(self) -> str:
        return "segment_true"

    def get_epoch_index(self, full_results_len: int) -> int:
        if full_results_len < 2:
            raise ValueError("Need at least 2 states for difference.")
        middle_epoch_plus_1 = full_results_len // 2
        epoch_index = middle_epoch_plus_1 - 1
        if epoch_index < 0:
            raise ValueError("Not enough epochs for middle calculation")
        self.logger.info(
            f"Calculating middle difference between epoch {epoch_index+1} and {epoch_index}"
        )
        return epoch_index


@InfluenceCalculatorFactory.register("true_last")
class TrueLastInfluenceCalculator(BaseDifferenceCalculator):
    # Note: Original 'true_last' function used 'length' arg but the helper didn't.
    # This implementation calculates the difference between the last two saved states.
    def _get_infl_type(self) -> str:
        return "true_last"

    def get_source_prefix(self) -> str:
        return "segment_true"

    def get_epoch_index(self, full_results_len: int) -> int:
        if full_results_len < 2:
            raise ValueError("Need at least 2 states for difference.")
        # Index required is num_states - 2 (to calculate state[last] - state[second_last])
        last_epoch_index = full_results_len - 2
        if last_epoch_index < 0:
            raise ValueError(
                "Not enough epochs for last calculation"
            )  # Should be covered by len < 2 check
        self.logger.info(
            f"Calculating last difference between epoch {last_epoch_index+1} and {last_epoch_index}"
        )
        return last_epoch_index


# --- TODO: Implement concrete classes for remaining types ---
# Examples: infl_lie, infl_icml, infl_tracin, segment_true, tim_all_epochs


@InfluenceCalculatorFactory.register("lie")
class LieInfluenceCalculator(InfluenceCalculator):
    """Computes LIE influence state at the end of each epoch."""

    def _get_infl_type(self) -> str:
        return "lie"

    # Override save method to save as 'lie_full'
    def _save(self, infl_data):
        """Helper to save results using the unified save_results function."""
        save_results(
            infl_data,
            self.dn,
            self.seed,
            "lie_full",
            self.logger,
            self.relabel_percentage,
        )
        if self.tb_writer:
            self.tb_writer.close()

    def _lie_helper(self, target_epoch: int) -> np.ndarray:
        """Calculates LIE influence integrated up to the end of target_epoch."""
        self.logger.debug(f"LIE Helper: Calculating state for epoch {target_epoch}")

        model_helper = get_network(
            self.model_type, self.input_dim, logger=self.logger
        ).to(self.device)

        # 1. Get u at the *end* of the target_epoch
        target_step_for_u = min(
            (target_epoch + 1) * self.steps_per_epoch, self.total_steps
        )
        try:
            step_data_end = load_step_data(
                self.dn,
                target_step_for_u,
                self.seed,
                self.relabel_percentage,
                self.logger,
            )
            model_helper.load_state_dict(step_data_end["model_state"])
            model_helper.eval()
            u = compute_gradient(self.x_val, self.y_val, model_helper, self.loss_fn)
            u = [uu.to(self.device) for uu in u]
            u = [
                uu.to(torch.float64 if u[0].dtype == torch.float64 else torch.float32)
            ]  # Match precision
            self.logger.debug(f"LIE Helper: Computed u from step {target_step_for_u}")
        except FileNotFoundError:
            self.logger.warning(
                f"LIE Helper: Step file {target_step_for_u} not found for epoch {target_epoch} u. Trying epoch file."
            )
            try:
                epoch_data_end = load_epoch_data(
                    self.dn,
                    target_epoch,
                    self.seed,
                    self.relabel_percentage,
                    self.logger,
                )
                model_helper.load_state_dict(epoch_data_end["model_state"])
                model_helper.eval()
                u = compute_gradient(self.x_val, self.y_val, model_helper, self.loss_fn)
                u = [
                    uu.to(self.device).to(
                        torch.float64 if u[0].dtype == torch.float64 else torch.float32
                    )
                    for uu in u
                ]
                self.logger.debug(
                    f"LIE Helper: Computed u from epoch {target_epoch} fallback file."
                )
            except FileNotFoundError:
                self.logger.warning(
                    f"LIE Helper: Epoch file {target_epoch} also not found. Using FINAL model u (less accurate)."
                )
                # Fallback to final model's gradient (less accurate for intermediate epochs)
                model_helper.load_state_dict(
                    load_final_model(
                        self.dn,
                        self.seed,
                        self.global_info,
                        self.fn_fallback,
                        self.device,
                        self.logger,
                        self.relabel_percentage,
                    )
                )
                model_helper.eval()
                u = compute_gradient(self.x_val, self.y_val, model_helper, self.loss_fn)
                u = [
                    uu.to(self.device).to(
                        torch.float64 if u[0].dtype == torch.float64 else torch.float32
                    )
                    for uu in u
                ]
        except Exception as e:
            self.logger.error(
                f"LIE Helper: Error getting u for epoch {target_epoch}: {e}",
                exc_info=True,
            )
            raise  # Or return zeros? Depends on desired behavior

        # 2. Reverse through steps up to the beginning of training (step 1)
        infl_epoch = np.zeros(self.n_tr, dtype=np.float64)
        start_step_incl = (
            target_step_for_u  # Start from state after last step of target epoch
        )
        end_step_excl = 0

        m_step = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )  # Reusable model

        for t in range(start_step_incl, end_step_excl, -1):
            step_log_prefix = f"LIE Helper Epoch {target_epoch} Step {t}"
            try:
                step_data = load_step_data(
                    self.dn, t, self.seed, self.relabel_percentage, self.logger
                )
                current_model_state = step_data["model_state"]
                idx, lr = step_data["idx"], step_data["lr"]

                m_step.load_state_dict(current_model_state)
                m_step.eval()

                if not isinstance(idx, (list, np.ndarray, torch.Tensor)):
                    idx = [idx]
                idx = torch.tensor(idx, device=self.device)
                if len(idx) == 0:
                    continue

                valid_idx_mask = (idx >= 0) & (idx < self.n_tr)
                idx = idx[valid_idx_mask]
                if len(idx) == 0:
                    continue

                batch_size = len(idx)
                x_batch, y_batch = self.x_tr[idx], self.y_tr[idx]

                # Accumulate influence (using per-sample grad loop)
                param_grads_list = []
                for i_local in range(batch_size):
                    m_step.zero_grad()
                    z_i = m_step(x_batch[[i_local]])
                    loss_i = self.loss_fn(z_i, y_batch[[i_local]])
                    if self.alpha > 0:
                        for p in m_step.parameters():
                            loss_i += 0.5 * self.alpha * (p * p).sum()
                    loss_i.backward()
                    grad_i = [
                        (
                            p.grad.data.clone().to(dtype=u[0].dtype)
                            if p.grad is not None
                            else torch.zeros_like(p, dtype=u[0].dtype)
                        )
                        for p in m_step.parameters()
                    ]
                    param_grads_list.append(grad_i)
                m_step.zero_grad()

                for i_local, sample_idx in enumerate(idx.tolist()):
                    grad_i = param_grads_list[i_local]
                    grad_sum = sum(
                        torch.sum(u[j].data * param_grad).item()
                        for j, param_grad in enumerate(grad_i)
                        if j < len(u)
                    )
                    infl_epoch[sample_idx] += lr * grad_sum / batch_size

                # Update u
                u_prev = [uu.clone() for uu in u]
                hvp = compute_hvp_with_finite_diff(
                    m_step, x_batch, y_batch, u, self.loss_fn, alpha=self.alpha
                )
                lambda_reg = compute_adaptive_lambda(
                    u_norm=sum_norm(u),
                    hu_norm=sum_norm(hvp),
                    base_lambda=0.1,
                    logger=self.logger,
                )
                hvp_reg = [
                    hv.to(dtype=u[0].dtype) + lambda_reg * uu for hv, uu in zip(hvp, u)
                ]

                for j in range(len(u)):
                    new_u_val = u[j] - lr * hvp_reg[j]
                    if torch.isnan(new_u_val).any() or torch.isinf(new_u_val).any():
                        u[j] = u_prev[j]  # Reset on instability
                    else:
                        u[j] = new_u_val  # Basic update for helper

            except FileNotFoundError:
                self.logger.warning(
                    f"{step_log_prefix}: Step file not found. Skipping."
                )
                continue
            except Exception as e:
                self.logger.error(f"{step_log_prefix}: Error: {e}", exc_info=True)
                continue  # Skip step on error

        self.logger.debug(
            f"LIE Helper: Finished epoch {target_epoch}, final u norm: {sum_norm(u):.4e}"
        )
        del m_step, model_helper
        gc.collect()
        torch.cuda.empty_cache()
        return infl_epoch

    def calculate(self) -> List[np.ndarray]:
        infl_list = (
            []
        )  # Stores influence calculated *up to* epoch 0, 1, ..., num_epochs
        # Calculate influence state at the end of each epoch (0 to num_epochs)
        # Epoch 0 means state after steps 1 to steps_per_epoch are done.
        for epoch in range(self.num_epochs + 1):  # Include state after final epoch
            self.logger.info(
                f"Calculating LIE influence state integrated up to epoch {epoch}"
            )
            infl = self._lie_helper(epoch)
            infl_list.append(infl)
            self.logger.info(f"Completed LIE influence state up to epoch {epoch}")

        # After saving the full list, optionally trigger the diff calculations
        # This might be better handled outside the main calculation loop
        # For now, just return the full list; saving happens in run() -> _save()
        return infl_list


import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim  # Added for IcmlInfluenceCalculator
from .NetworkModule import (
    get_network,
    NetList,
)  # Added NetList for fallback loading check
from .logging_utils import (
    setup_logging,
)  # Make sure this import is correct relative to structure
from .vis import sum_norm  # Import sum_norm if needed
from .DataModule import fetch_data_module  # Make sure this import is correct
import logging
import gc
from typing import List, Dict, Any, Union

# --- Assumed Helper Functions (already defined in the main script) ---
# get_device, get_file_paths, load_global_info, load_data, get_input_dim,
# compute_gradient, load_epoch_data, load_step_data, load_initial_model, load_final_model,
# save_results, compute_adaptive_lambda, compute_hvp_with_finite_diff,
# InfluenceCalculator, InfluenceCalculatorFactory, BaseDifferenceCalculator

# --- Constants for ICML (from the provided code) ---
BATCH_SIZE_ICML = 200
LR_ICML = 0.01
MOMENTUM_ICML = 0.9
NUM_EPOCHS_ICML = 100

# --- Concrete Calculator Implementations ---


@InfluenceCalculatorFactory.register("icml")
class IcmlInfluenceCalculator(InfluenceCalculator):
    """
    Computes influence using the ICML'17 method (Hessian inversion approximation via optimization).
    Approximates H^-1 v by optimizing a surrogate objective.
    """

    def _get_infl_type(self) -> str:
        return "icml"

    def _save(self, result_tuple):
        """Override save to handle influence array and training loss."""
        infl_data, loss_train = result_tuple
        # Save main influence results using base save_results
        save_results(
            infl_data,
            self.dn,
            self.seed,
            self.infl_type,  # Should be "icml"
            self.logger,
            self.relabel_percentage,
        )
        # Save the auxiliary training loss data
        hn = os.path.join(self.dn, f"loss_icml{self.seed:03d}.dat")
        try:
            # Convert to numpy array before saving if it's a list
            loss_train_np = np.array(loss_train)
            torch.save(loss_train_np, hn)
            self.logger.info(f"ICML optimization training loss saved to {hn}")
        except Exception as e:
            self.logger.error(f"Failed to save ICML training loss to {hn}: {e}")

        if self.tb_writer:
            self.tb_writer.close()  # Close writer after saving

    def calculate(self) -> tuple[np.ndarray, list]:
        # Get dynamic Hessian regularization parameters, default is True
        use_dynamic_hessian_reg = self.kwargs.get("dynamic_hessian_reg", True)
        hessian_reg_base = self.kwargs.get("hessian_reg_base", 0.01)
        hessian_reg_max = self.kwargs.get("hessian_reg_max", 10.0)
        hessian_norm_threshold = self.kwargs.get("hessian_norm_threshold", 1e5)

        if use_dynamic_hessian_reg:
            self.logger.info(f"Dynamic Hessian regularization enabled: base={hessian_reg_base}, max={hessian_reg_max}, norm threshold={hessian_norm_threshold}")

        self.logger.info("Starting ICML influence calculation...")

        # --- Load Final Model ---
        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        model.load_state_dict(
            load_final_model(
                self.dn,
                self.seed,
                self.global_info,
                self.fn_fallback,
                self.device,
                self.logger,
                self.relabel_percentage,
            )
        )
        model.eval()
        self.logger.info("Final model loaded for ICML.")

        # --- Compute u = grad(L_val) at final parameters ---
        u = compute_gradient(self.x_val, self.y_val, model, self.loss_fn)
        u = [
            uu.to(self.device) for uu in u
        ]
        self.logger.info("Computed initial gradient 'u' from validation set.")

        # --- Optimization to find v ≈ H^-1 u ---
        alpha_icml = self.alpha
        num_steps_icml = int(np.ceil(self.n_tr / BATCH_SIZE_ICML))
        v = [uu.clone().detach().requires_grad_(True) for uu in u]
        optimizer = optim.SGD(v, lr=LR_ICML, momentum=MOMENTUM_ICML)
        loss_train_icml = []

        self.logger.info(
            f"Starting H^-1 v optimization ({NUM_EPOCHS_ICML} epochs, {num_steps_icml} steps/epoch)"
        )
        for epoch in range(NUM_EPOCHS_ICML):
            model.eval()
            np.random.seed(epoch)
            idx_list = np.array_split(np.random.permutation(self.n_tr), num_steps_icml)

            epoch_loss_sum = 0.0

            # If dynamic regularization is enabled, calculate current v norm and adjust regularization coefficient
            current_hessian_reg = hessian_reg_base
            if use_dynamic_hessian_reg:
                current_v_norm = sum_norm(v)
                if current_v_norm > hessian_norm_threshold:
                    # Convert tensors to CPU scalars before using with numpy
                    norm_ratio = (current_v_norm / hessian_norm_threshold).item()
                    current_hessian_reg = min(
                        hessian_reg_base * (1.0 + np.log10(norm_ratio)), 
                        hessian_reg_max
                    )
                    self.logger.info(
                        f"Epoch {epoch+1}: v_norm={current_v_norm:.4e}, dynamic Hessian reg={current_hessian_reg:.6f}"
                    )

            for i, idx_batch in enumerate(idx_list):
                idx_tensor = torch.tensor(
                    idx_batch, dtype=torch.long, device=self.device
                )
                x_batch = self.x_tr[idx_tensor]
                y_batch = self.y_tr[idx_tensor]

                z = model(x_batch)
                loss = self.loss_fn(z, y_batch)

                model.zero_grad()
                grad_params = torch.autograd.grad(
                    loss, model.parameters(), create_graph=True
                )

                vg = sum((vv * g).sum() for vv, g in zip(v, grad_params))

                vgrad_params = torch.autograd.grad(
                    vg, model.parameters(), create_graph=False
                )

                # Apply dynamic Hessian regularization (H_reg = H + lambda * I)
                if use_dynamic_hessian_reg:
                    # The original loss: 0.5 * v^T H v - u^T v + 0.5 * alpha * v^T v
                    # With Hessian regularization: 0.5 * v^T (H + lambda * I) v - u^T v + 0.5 * alpha * v^T v
                    # This is equivalent to adding 0.5 * lambda * v^T v
                    loss_i = sum(
                        0.5 * (vgp * vv + (alpha_icml + current_hessian_reg) * vv * vv).sum() - (uu * vv).sum()
                        for vgp, vv, uu in zip(vgrad_params, v, u)
                    )
                else:
                    # Original loss calculation
                    loss_i = sum(
                        0.5 * (vgp * vv + alpha_icml * vv * vv).sum() - (uu * vv).sum()
                        for vgp, vv, uu in zip(vgrad_params, v, u)
                    )

                optimizer.zero_grad()
                loss_i.backward()
                optimizer.step()

                # After optimization step, add norm clipping to prevent numerical instability
                if use_dynamic_hessian_reg:
                    with torch.no_grad():
                        v_norm = sum_norm(v)
                        if v_norm > 1e15:  # Set a reasonable threshold to prevent overflow
                            scale_factor = 1e15 / v_norm
                            for idx in range(len(v)):
                                v[idx].mul_(scale_factor)
                            self.logger.warning(
                                f"Epoch {epoch+1}, Step {i}: Clipped v norm from {v_norm:.4e} to {1e15:.4e}"
                            )

                current_loss_val = loss_i.item()
                loss_train_icml.append(current_loss_val)
                epoch_loss_sum += current_loss_val

                # TensorBoard Logging for optimization
                if self.tb_writer is not None:
                    global_step = epoch * num_steps_icml + i
                    self.tb_writer.add_scalar(
                        f"{self.infl_type}/optim_loss", current_loss_val, global_step
                    )
                    if i == 0:  # Log v norm once per epoch
                        self.tb_writer.add_scalar(
                            f"{self.infl_type}/v_norm", sum_norm(v), global_step
                        )
                        if use_dynamic_hessian_reg:
                            self.tb_writer.add_scalar(
                                f"{self.infl_type}/hessian_reg", current_hessian_reg, global_step
                            )

            avg_epoch_loss = epoch_loss_sum / num_steps_icml
            self.logger.info(
                f"Epoch {epoch+1}/{NUM_EPOCHS_ICML}, Avg. Optimization Loss: {avg_epoch_loss:.6f}, Final v norm: {sum_norm(v):.4e}"
            )

        self.logger.info("Optimization finished. Calculating final influence scores.")
        # --- Final Influence Calculation ---
        # infl_i = - grad(L_train_i) dot v (H^-1 u approximation) / n_tr
        infl = np.zeros(self.n_tr, dtype=np.float64)
        # Process per sample (can be slow, consider batching later if needed)
        for i in range(self.n_tr):
            x_i = self.x_tr[[i]]
            y_i = self.y_tr[[i]]

            z = model(x_i)
            loss = self.loss_fn(z, y_i)
            # Add L2 regularization term to the loss *of the training sample*?
            # Original code doesn't seem to add L2 here for the final dot product.
            # Let's compute grad(L_train_i) without L2 for consistency with original snippet.
            # if self.alpha > 0:
            #    l2_reg = 0.0
            #    for p in model.parameters():
            #        l2_reg += 0.5 * self.alpha * torch.sum(p * p)
            #    loss += l2_reg

            model.zero_grad()
            loss.backward()  # Compute gradients of loss w.r.t model parameters

            # Calculate dot product: grad(L_train_i) dot v
            infl_i = 0.0
            with torch.no_grad():  # Don't need gradients for this part
                for j, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        # Ensure dtypes match for dot product, use float64 if v is float64
                        param_grad_device = param.grad.data.to(
                            v[j].device, dtype=v[j].dtype
                        )
                        infl_i += torch.sum(param_grad_device * v[j].data).item()
                    else:
                        # Handle case where a parameter might not have a gradient (e.g., unused)
                        self.logger.debug(
                            f"Parameter {j} has no gradient for sample {i}."
                        )
                        pass  # infl_i contribution is zero

            # Store influence, normalize by n_tr
            # The negative sign is standard for influence functions definition.
            infl[i] = -infl_i / self.n_tr

            if (i + 1) % 500 == 0:
                self.logger.info(
                    f"Calculated ICML influence for {i+1}/{self.n_tr} samples."
                )

        # Clean up optimizer and v gradients
        del optimizer, v
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return both influence and training loss history
        return infl, loss_train_icml


@InfluenceCalculatorFactory.register("tracin")
class TracinInfluenceCalculator(InfluenceCalculator):
    """
    Computes TracIn influence scores by summing gradient dot products over checkpoints.
    Influence(z_train, z_test) = sum_{k=checkpoints} lr_k * grad(L(theta_k, z_train)) dot grad(L(theta_k, z_test))
    Here, z_test corresponds to the validation set (x_val, y_val).
    """

    def _get_infl_type(self) -> str:
        return "tracin"

    # Override _save to include CSV saving logic from original function
    def _save(self, infl_data):
        """Save .dat and detailed .csv results."""
        # Save .dat using base save_results
        save_results(
            infl_data,
            self.dn,
            self.seed,
            self.infl_type,  # Should be "tracin"
            self.logger,
            self.relabel_percentage,
        )

        # Save detailed CSV
        csv_fn = os.path.join(self.dn, f"infl_{self.infl_type}_{self.seed:03d}.csv")
        # Adjust filename if relabeled
        if self.relabel_percentage is not None:
            relabel_prefix = f"relabel_{int(self.relabel_percentage):03d}_pct_"
            csv_fn = os.path.join(
                self.dn, f"infl_{self.infl_type}_{relabel_prefix}{self.seed:03d}.csv"
            )

        try:
            # Need y_tr for the CSV which is loaded in common setup
            y_tr_np = self.y_tr.cpu().numpy().flatten()  # Get numpy array of labels

            df = pd.DataFrame(
                {
                    "sample_idx": np.arange(self.n_tr),
                    "true_label": y_tr_np,
                    "influence": infl_data,  # Already a numpy array
                    "influence_rank": pd.Series(infl_data)
                    .rank(ascending=False, method="first")
                    .astype(int),
                    "influence_percentile": pd.Series(infl_data).rank(pct=True),
                }
            )
            df.to_csv(csv_fn, index=False)
            self.logger.info(f"TracIn detailed results saved to CSV: {csv_fn}")
        except Exception as e:
            self.logger.error(f"Failed to save TracIn results to CSV {csv_fn}: {e}")

        if self.tb_writer:
            self.tb_writer.close()  # Close writer after saving

    def calculate(self) -> np.ndarray:
        self.logger.info("Starting TracIn influence calculation...")

        # Determine checkpoints (e.g., 30%, 60%, 90% of training steps)
        # Use total_steps calculated in _setup_common
        checkpoints = [
            int(self.total_steps * 0.3),
            int(self.total_steps * 0.6),
            int(self.total_steps * 0.9),  # Use 0.9, the -1 seemed arbitrary
        ]
        # Ensure checkpoints are valid indices (>=1, <= total_steps)
        checkpoints = [max(1, min(cp, self.total_steps)) for cp in checkpoints]
        checkpoints = sorted(list(set(checkpoints)))  # Ensure unique and sorted, >= 1

        self.logger.info(
            f"Using checkpoints (steps): {checkpoints} out of {self.total_steps}"
        )

        # Initialize influence scores
        infl = np.zeros(self.n_tr, dtype=np.float64)

        # Temporary model instance
        m_checkpoint = get_network(
            self.model_type, self.input_dim, logger=self.logger
        ).to(self.device)

        # Loop through selected checkpoints
        for step_k in checkpoints:
            self.logger.info(f"--- Processing Checkpoint: Step {step_k} ---")
            # Load model state and learning rate for this checkpoint
            try:
                step_data = load_step_data(
                    self.dn, step_k, self.seed, self.relabel_percentage, self.logger
                )
                m_checkpoint.load_state_dict(step_data["model_state"])
                lr_k = step_data["lr"]
                self.logger.info(f"Loaded state and lr={lr_k:.6f} from step {step_k}")
            except FileNotFoundError:
                self.logger.warning(
                    f"Step file for step {step_k} not found. Trying epoch fallback."
                )
                try:
                    epoch_k = (step_k - 1) // self.steps_per_epoch
                    epoch_data = load_epoch_data(
                        self.dn,
                        epoch_k,
                        self.seed,
                        self.relabel_percentage,
                        self.logger,
                    )
                    m_checkpoint.load_state_dict(epoch_data["model_state"])
                    # Try to find specific step info within epoch data
                    step_in_epoch_idx = (step_k - 1) % self.steps_per_epoch
                    if "step_info" in epoch_data and step_in_epoch_idx < len(
                        epoch_data["step_info"]
                    ):
                        lr_k = epoch_data["step_info"][step_in_epoch_idx]["lr"]
                        self.logger.info(
                            f"Loaded state from epoch {epoch_k}, lr={lr_k:.6f} from step info"
                        )
                    else:
                        lr_k = self.global_info.get("lr", 0.01)  # Fallback LR
                        self.logger.warning(
                            f"Loaded state from epoch {epoch_k}, step info for lr not found. Using global lr={lr_k:.6f}"
                        )
                except FileNotFoundError:
                    self.logger.error(
                        f"Epoch file for epoch {epoch_k} also not found. Skipping checkpoint {step_k}."
                    )
                    continue
                except Exception as e:
                    self.logger.error(
                        f"Error loading epoch fallback for step {step_k}: {e}. Skipping checkpoint."
                    )
                    continue
            except Exception as e:
                self.logger.error(
                    f"Error loading step data for checkpoint {step_k}: {e}. Skipping checkpoint."
                )
                continue

            m_checkpoint.eval()

            # Compute target gradient: grad(L_val) at checkpoint k
            # We compute gradient w.r.t ALL validation samples together
            m_checkpoint.zero_grad()
            z_val = m_checkpoint(self.x_val)
            loss_val = self.loss_fn(z_val, self.y_val)
            # Add L2 regularization? TracIn paper doesn't explicitly mention adding L2 to test loss grad. Let's omit it.
            # if self.alpha > 0:
            #    l2_reg = 0.5 * self.alpha * sum(p.pow(2).sum() for p in m_checkpoint.parameters())
            #    loss_val += l2_reg
            loss_val.backward()
            grad_val = [
                (
                    p.grad.data.clone().detach()
                    if p.grad is not None
                    else torch.zeros_like(p)
                )
                for p in m_checkpoint.parameters()
            ]
            self.logger.info(
                f"Computed validation gradient at step {step_k}. Norm: {sum_norm(grad_val):.4e}"
            )

            # Loop through training samples (consider batching this loop for performance)
            batch_size_tracin = 512  # Batch size for processing training gradients
            num_tr_batches = (self.n_tr + batch_size_tracin - 1) // batch_size_tracin

            for i_batch in range(num_tr_batches):
                start_idx = i_batch * batch_size_tracin
                end_idx = min(start_idx + batch_size_tracin, self.n_tr)
                x_tr_batch = self.x_tr[start_idx:end_idx]
                y_tr_batch = self.y_tr[start_idx:end_idx]

                # Compute gradients for the training batch samples individually
                for i_local in range(end_idx - start_idx):
                    idx_global = start_idx + i_local
                    x_i = x_tr_batch[[i_local]]
                    y_i = y_tr_batch[[i_local]]

                    m_checkpoint.zero_grad()
                    z_tr_i = m_checkpoint(x_i)
                    loss_tr_i = self.loss_fn(z_tr_i, y_i)
                    # Add L2 regularization to training loss gradient? Consistent with SGD run? Yes.
                    if self.alpha > 0:
                        l2_reg = 0.0
                        for p in m_checkpoint.parameters():
                            l2_reg += 0.5 * self.alpha * torch.sum(p * p)
                        loss_tr_i += l2_reg

                    loss_tr_i.backward()
                    grad_tr_i = [
                        (
                            p.grad.data.clone().detach()
                            if p.grad is not None
                            else torch.zeros_like(p)
                        )
                        for p in m_checkpoint.parameters()
                    ]

                    # Calculate dot product and accumulate influence
                    dot_prod = 0.0
                    with torch.no_grad():
                        for g_tr, g_val in zip(grad_tr_i, grad_val):
                            dot_prod += torch.sum(
                                g_tr.to(dtype=g_val.dtype) * g_val
                            ).item()  # Ensure dtype match

                    infl[idx_global] += lr_k * dot_prod

                if (i_batch + 1) % 20 == 0:  # Log progress periodically
                    self.logger.info(
                        f"Checkpoint {step_k}: Processed training batch {i_batch+1}/{num_tr_batches}"
                    )

            # Clear cache after processing a checkpoint
            del grad_val, step_data  # Potentially large objects
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info(f"--- Finished Checkpoint: Step {step_k} ---")

        del m_checkpoint  # Clean up model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("TracIn calculation finished.")
        return infl  # Return the final accumulated influence scores


@InfluenceCalculatorFactory.register("tim_all_epochs")
class TimAllEpochsInfluenceCalculator(InfluenceCalculator):
    """
    Computes influence for each epoch interval using the reverse SGD method.
    For epoch k, it reverses from step (k+1)*SPE back to k*SPE + 1,
    using the validation gradient computed at step (k+1)*SPE as the initial 'u'.
    """

    def _get_infl_type(self) -> str:
        return "tim_all_epochs"

    # Saving handled by base class _save and save_results (which supports lists)

    def calculate(self) -> List[np.ndarray]:
        self.logger.info("Starting TIM All Epochs influence calculation...")

        all_epoch_infl = []

        # Temporary model instance for loading states
        m_step = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )

        # Loop through each epoch from 0 to num_epochs - 1
        for epoch_idx in range(self.num_epochs):
            self.logger.info(f"--- Calculating Influence for Epoch {epoch_idx} ---")
            infl_epoch = np.zeros(
                self.n_tr, dtype=np.float64
            )  # Accumulator for this epoch

            # Determine step range for this epoch's reversal
            # Start reversing *from* the state AFTER the last step of this epoch
            start_step_incl = min(
                (epoch_idx + 1) * self.steps_per_epoch, self.total_steps
            )
            # Stop reversing *before* the first step of this epoch (or step 1 if epoch 0)
            end_step_excl = epoch_idx * self.steps_per_epoch

            if start_step_incl <= end_step_excl:
                self.logger.warning(
                    f"Epoch {epoch_idx}: Start step {start_step_incl} <= end step {end_step_excl}. Skipping epoch."
                )
                all_epoch_infl.append(infl_epoch)  # Append zeros
                continue

            self.logger.info(
                f"Epoch {epoch_idx}: Reversing from step {start_step_incl} down to {end_step_excl + 1}"
            )

            # 1. Get u_epoch_end: Gradient of validation loss at the END of this epoch (start_step_incl)
            u_epoch_end = None
            try:
                step_data_end = load_step_data(
                    self.dn,
                    start_step_incl,
                    self.seed,
                    self.relabel_percentage,
                    self.logger,
                )
                m_step.load_state_dict(step_data_end["model_state"])
                m_step.eval()
                u_epoch_end = compute_gradient(
                    self.x_val, self.y_val, m_step, self.loss_fn
                )
                u_epoch_end = [uu.to(self.device) for uu in u_epoch_end]
                try:  # Use float64 if possible
                    u_epoch_end = [uu.to(torch.float64) for uu in u_epoch_end]
                    u_dtype = torch.float64
                except TypeError:
                    u_epoch_end = [uu.to(torch.float32) for uu in u_epoch_end]
                    u_dtype = torch.float32
                self.logger.info(
                    f"Epoch {epoch_idx}: Computed u_epoch_end from step {start_step_incl}. Norm: {sum_norm(u_epoch_end):.4e}"
                )
            except FileNotFoundError:
                self.logger.warning(
                    f"Epoch {epoch_idx}: Step file {start_step_incl} not found for u_epoch_end. Trying epoch file."
                )
                try:
                    # Load from epoch file corresponding to the END of epoch_idx
                    epoch_data_end = load_epoch_data(
                        self.dn,
                        epoch_idx,
                        self.seed,
                        self.relabel_percentage,
                        self.logger,
                    )
                    m_step.load_state_dict(epoch_data_end["model_state"])
                    m_step.eval()
                    u_epoch_end = compute_gradient(
                        self.x_val, self.y_val, m_step, self.loss_fn
                    )
                    u_epoch_end = [uu.to(self.device) for uu in u_epoch_end]
                    try:  # Use float64 if possible
                        u_epoch_end = [uu.to(torch.float64) for uu in u_epoch_end]
                        u_dtype = torch.float64
                    except TypeError:
                        u_epoch_end = [uu.to(torch.float32) for uu in u_epoch_end]
                        u_dtype = torch.float32
                    self.logger.info(
                        f"Epoch {epoch_idx}: Computed u_epoch_end using epoch file {epoch_idx} fallback. Norm: {sum_norm(u_epoch_end):.4e}"
                    )
                except FileNotFoundError:
                    self.logger.error(
                        f"Epoch {epoch_idx}: Cannot load step or epoch data for state after epoch. Skipping epoch."
                    )
                    all_epoch_infl.append(infl_epoch)  # Append zeros
                    continue
                except Exception as e:
                    self.logger.error(
                        f"Epoch {epoch_idx}: Error loading epoch fallback for u_epoch_end: {e}. Skipping epoch."
                    )
                    all_epoch_infl.append(infl_epoch)
                    continue
            except Exception as e:
                self.logger.error(
                    f"Epoch {epoch_idx}: Error loading step data for u_epoch_end: {e}. Skipping epoch."
                )
                all_epoch_infl.append(infl_epoch)
                continue

            # Clone u_epoch_end to use for this segment's reverse pass
            u_current_segment = [ue.clone() for ue in u_epoch_end]

            # 2. Reverse through steps within this epoch
            for t in range(start_step_incl, end_step_excl, -1):
                step_log_prefix = f"Epoch {epoch_idx} Step {t}"
                try:
                    # Load data for state *before* step t (state at step t)
                    step_data = load_step_data(
                        self.dn, t, self.seed, self.relabel_percentage, self.logger
                    )
                    m_step.load_state_dict(step_data["model_state"])
                    m_step.eval()
                    idx, lr = step_data["idx"], step_data["lr"]

                    if not isinstance(idx, (list, np.ndarray, torch.Tensor)):
                        idx = [idx]
                    idx = torch.tensor(idx, device=self.device)  # Ensure tensor
                    if len(idx) == 0:
                        continue

                    # Filter invalid indices (safeguard)
                    valid_idx_mask = (idx >= 0) & (idx < self.n_tr)
                    idx = idx[valid_idx_mask]
                    if len(idx) == 0:
                        continue

                    batch_size = len(idx)
                    x_batch, y_batch = self.x_tr[idx], self.y_tr[idx]

                    # a) Accumulate influence using u_current_segment
                    # Use per-sample grad loop for accuracy
                    param_grads_list = []
                    for i_local in range(batch_size):
                        m_step.zero_grad()
                        z_i = m_step(x_batch[[i_local]])
                        loss_i = self.loss_fn(z_i, y_batch[[i_local]])
                        if self.alpha > 0:
                            for p in m_step.parameters():
                                loss_i += 0.5 * self.alpha * (p * p).sum()
                        loss_i.backward()
                        grad_i = [
                            (
                                p.grad.data.clone().to(dtype=u_dtype)
                                if p.grad is not None
                                else torch.zeros_like(p, dtype=u_dtype)
                            )
                            for p in m_step.parameters()
                        ]
                        param_grads_list.append(grad_i)
                    m_step.zero_grad()  # Clean up

                    for i_local, sample_idx in enumerate(idx.tolist()):
                        grad_i = param_grads_list[i_local]
                        grad_sum = 0.0
                        for j, param_grad in enumerate(grad_i):
                            if j < len(u_current_segment):
                                grad_sum += torch.sum(
                                    u_current_segment[j].data * param_grad
                                ).item()
                        infl_epoch[sample_idx] += (
                            lr * grad_sum / batch_size
                        )  # Accumulate for this epoch

                    # b) Update u_current_segment using HVP
                    u_prev = [uu.clone() for uu in u_current_segment]
                    hvp = compute_hvp_with_finite_diff(
                        m_step,
                        x_batch,
                        y_batch,
                        u_current_segment,
                        self.loss_fn,
                        alpha=self.alpha,
                    )  # Pass alpha
                    lambda_reg = compute_adaptive_lambda(
                        u_norm=sum_norm(u_current_segment),
                        hu_norm=sum_norm(hvp),
                        base_lambda=0.1,
                        logger=self.logger,
                    )
                    hvp_reg = [
                        hv.to(dtype=u_dtype) + lambda_reg * uu
                        for hv, uu in zip(hvp, u_current_segment)
                    ]

                    for j in range(len(u_current_segment)):
                        new_u_val = u_current_segment[j] - lr * hvp_reg[j]
                        if torch.isnan(new_u_val).any() or torch.isinf(new_u_val).any():
                            self.logger.warning(
                                f"{step_log_prefix}: NaN/Inf in u_current_segment[{j}] update, resetting."
                            )
                            u_current_segment[j] = u_prev[j]
                        else:
                            # Apply clipping based on norm increase? (Optional stability)
                            old_norm = u_current_segment[j].norm().item()
                            new_norm = new_u_val.norm().item()
                            if old_norm > 1e-9 and new_norm / old_norm > 100.0:
                                scale = 100.0 * old_norm / new_norm
                                u_current_segment[j] = (
                                    u_current_segment[j] - scale * lr * hvp_reg[j]
                                )
                                self.logger.warning(
                                    f"{step_log_prefix}: Clipped u_current_segment[{j}] update, norm ratio {new_norm/old_norm:.2f}"
                                )
                            else:
                                u_current_segment[j] = new_u_val

                    # Log progress within epoch reversal
                    if (start_step_incl - t) % 50 == 0:
                        self.logger.info(
                            f"{step_log_prefix} processed. u_segment norm: {sum_norm(u_current_segment):.4e}"
                        )
                        if self.tb_writer:
                            # Log relative step within this calculation pass
                            relative_step = start_step_incl - t
                            self.tb_writer.add_scalar(
                                f"{self.infl_type}/epoch_{epoch_idx}/u_norm",
                                sum_norm(u_current_segment),
                                relative_step,
                            )
                            self.tb_writer.add_scalar(
                                f"{self.infl_type}/epoch_{epoch_idx}/lambda_reg",
                                lambda_reg,
                                relative_step,
                            )

                except FileNotFoundError:
                    # Attempt epoch fallback for the specific step t state (less ideal for reverse)
                    self.logger.warning(
                        f"{step_log_prefix}: Step file not found. Trying epoch fallback (may be inaccurate)."
                    )
                    # TODO: Implement epoch fallback similar to SgdInfluenceCalculator if needed,
                    # but this might use a stale model state for the HVP and gradient calculations.
                    # For now, skipping the step if file is missing.
                    self.logger.error(
                        f"{step_log_prefix}: Step file missing and epoch fallback not implemented here. Skipping step."
                    )
                    continue
                except Exception as e_outer:
                    self.logger.error(
                        f"{step_log_prefix}: Error processing step: {e_outer}",
                        exc_info=True,
                    )
                    continue  # Skip this step

                # Optional: Clear GPU cache periodically within epoch reversal
                if (start_step_incl - t) % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # After reversing all steps for this epoch, store the accumulated influence
            all_epoch_infl.append(infl_epoch)
            self.logger.info(
                f"--- Finished Influence Calculation for Epoch {epoch_idx} ---"
            )

            # Clean up between epochs
            del u_epoch_end, u_current_segment
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del m_step  # Clean up model instance
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("TIM All Epochs calculation finished.")
        return all_epoch_infl


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Compute Influence Functions (Factory Pattern)"
    )
    parser.add_argument(
        "--target", default="adult", type=str, help="Target dataset key"
    )
    parser.add_argument("--model", default="logreg", type=str, help="Model type key")
    parser.add_argument(
        "--type", default="sgd", type=str, help="Influence calculation type"
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed (>= 0 for single run, < 0 for loop 0-99)",
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU index")
    parser.add_argument(
        "--save_dir", type=str, help="Optional directory override for saving results"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--relabel",
        type=float,
        help="Percentage of training data to relabel (e.g., 10 for 10%)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=3,
        help="Length parameter for tim_last influence computation",
    )
    parser.add_argument(
        "--use_tensorboard", action="store_true", help="Enable TensorBoard logging"
    )

    args = parser.parse_args()

    # Set root logger level first
    try:
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, args.log_level.upper()))
    except AttributeError:
        logging.warning(f"Invalid log level '{args.log_level}'. Defaulting to INFO.")
        logging.getLogger().setLevel(logging.INFO)

    logger_main = logging.getLogger("main")  # Specific logger for main function

    # Validate essential registry lookups early
    if args.target not in DATA_MODULE_REGISTRY:
        logger_main.error(
            f"Invalid target data '{args.target}'. Choose from {list(DATA_MODULE_REGISTRY.keys())}."
        )
        return  # Exit early
    if args.model not in NETWORK_REGISTRY:
        logger_main.error(
            f"Invalid model type '{args.model}'. Choose from {list(NETWORK_REGISTRY.keys())}."
        )
        return  # Exit early

    # Prepare arguments for the factory/calculator __init__
    # Pass all relevant args parsed; the calculator's __init__ will pick what it needs.
    calculator_args = {
        "key": args.target,
        "model_type": args.model,
        "seed": args.seed,
        "gpu": args.gpu,
        "save_dir": args.save_dir,  # This is the override path
        "relabel_percentage": args.relabel,
        "use_tensorboard": args.use_tensorboard,
        "length": args.length,  # Pass length; calculators that don't need it will ignore it via **kwargs
        # Add other specific args here if needed by future calculators
    }

    # --- Run Calculation ---
    if args.seed >= 0:
        logger_main.info(f"--- Running Influence Calculation ---")
        logger_main.info(
            f"Type: {args.type}, Dataset: {args.target}, Model: {args.model}, Seed: {args.seed}"
        )
        try:
            # Create and run the calculator
            calculator = InfluenceCalculatorFactory.create(args.type, **calculator_args)
            calculator.run()  # run() handles calculation and saving
            logger_main.info(f"--- Calculation successful for seed {args.seed} ---")
        except ValueError as e:  # Catch factory errors (unknown type)
            logger_main.error(f"Configuration error: {e}")
        except NotImplementedError as e:
            logger_main.error(
                f"Calculation type '{args.type}' is registered but not fully implemented: {e}"
            )
        except Exception as e:
            logger_main.error(
                f"Calculation failed for seed {args.seed}: {e}", exc_info=True
            )  # Log traceback
    else:
        # Loop over seeds if seed is negative
        logger_main.info(
            f"Seed < 0 detected. Running for seeds 0 to 99 for type '{args.type}'..."
        )
        successful_seeds = 0
        failed_seeds = []
        for seed_val in range(100):
            logger_main.info(f"--- Running for Seed {seed_val} ---")
            current_calculator_args = calculator_args.copy()
            current_calculator_args["seed"] = seed_val
            try:
                calculator = InfluenceCalculatorFactory.create(
                    args.type, **current_calculator_args
                )
                calculator.run()
                successful_seeds += 1
                logger_main.info(f"--- Calculation successful for seed {seed_val} ---")
            except ValueError as e:
                logger_main.error(f"[Seed {seed_val}] Configuration error: {e}")
                failed_seeds.append(seed_val)
            except NotImplementedError as e:
                logger_main.error(
                    f"[Seed {seed_val}] Calculation type '{args.type}' not fully implemented: {e}"
                )
                failed_seeds.append(seed_val)
            except Exception as e:
                logger_main.error(
                    f"[Seed {seed_val}] Calculation failed: {e}", exc_info=True
                )
                failed_seeds.append(seed_val)
            finally:
                # Optional: Force cleanup between seeds if memory issues occur
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger_main.info(f"--- Seed Loop Summary ---")
        logger_main.info(f"Successfully completed seeds: {successful_seeds}/100")
        if failed_seeds:
            logger_main.warning(f"Failed seeds: {failed_seeds}")
        else:
            logger_main.info("All seeds completed without errors.")


if __name__ == "__main__":
    # Note: Ensure the dependent modules (DataModule, NetworkModule, etc.)
    # are correctly structured relative to this script for the imports to work.
    # Example: If this script is 'run_influence.py', the structure might be:
    # my_project/
    #   run_influence.py
    #   DataModule/
    #       __init__.py
    #       ...
    #   NetworkModule/
    #       __init__.py
    #       ...
    #   logging_utils.py
    #   vis.py

    # Make sure imports reflect the actual structure. If running as a script,
    # you might need to adjust imports (e.g., from DataModule import ...)
    # or run using `python -m my_project.run_influence --args...`

    main()
