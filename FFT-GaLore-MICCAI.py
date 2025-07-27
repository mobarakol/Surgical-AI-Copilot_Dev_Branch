#---------------------------Import Packages------------------------------------------------------
import torch
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import logging
import glob
import argparse
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
import torch.nn as nn

import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version


from huggingface_hub import login
login(token="my_token")

# ------------------------- Argument Parsing -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with GaLore")
    # Model and tokenizer configuration
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Pre-trained model name or path")
    # Data paths
    parser.add_argument("--train_file", type=str, default="/SAN/medic/Surgical_LLM_Agent/11Respawn/Dataset/Train/Train.csv", help="Path to the training data CSV")
    parser.add_argument("--val_file", type=str, default="/SAN/medic/Surgical_LLM_Agent/11Respawn/Dataset/Test/Test.csv", help="Path to the validation data CSV")
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-7, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8, help="The rank of the update matrices in LoRA")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.3, help="LoRA dropout probability")
    # GaLore Configuration
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="reverse_std")
    # Model save path
    parser.add_argument("--save_path", type=str, default="/SAN/medic/Surgical_LLM_Agent/11Respawn/Best-Models/FFT-GaLore/3e-7/r128/10-epoch", help="Path to save the best model")
    # Other parameters
    parser.add_argument("--seed", type=int, default=50, help="Random seed")
    # New parameters
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--use_chat_template", action='store_true', help="Whether to use chat template to format input")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Maximum number of new tokens to generate")
    
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Base pre-trained model name or path")
    parser.add_argument("--best_model_path", type=str, default="/SAN/medic/Surgical_LLM_Agent/11Respawn/Best-Models/FFT-GaLore/3e-7/r128/10-epoch",
                        help="Path to the best fine-tuned model")
    parser.add_argument("--input_files", type=str,
                        default="/SAN/medic/Surgical_LLM_Agent/11Respawn/Dataset/Test/Surgical-VQA_V.csv,"
                                "/SAN/medic/Surgical_LLM_Agent/11Respawn/Dataset/Test/Segment-MRI_V.csv,"
                                "/SAN/medic/Surgical_LLM_Agent/11Respawn/Dataset/Test/Segment-Video_V.csv,"
                                "/SAN/medic/Surgical_LLM_Agent/11Respawn/Dataset/Test/Track-Instrument_V.csv,"
                                "/SAN/medic/Surgical_LLM_Agent/11Respawn/Dataset/Test/2-model_V.csv,"
                                "/SAN/medic/Surgical_LLM_Agent/11Respawn/Dataset/Test/3-model_V.csv",
                        help="Comma-separated list of evaluation file paths")
    parser.add_argument("--output_dir", type=str, default="/SAN/medic/Surgical_LLM_Agent/11Respawn/Results/FFT-GaLore/3e-7/r128/10-epoch",
                        help="Output directory to save evaluation results")
    parser.add_argument("--lr_description", type=str, default="3e-7", help="Learning rate description")
    return parser.parse_args()

#---------------------------FFT-Galore part-------------------------------------------

def fft_projection(A, k):
        # Step 1: FFT along the columns (dim=1)
        matrix_fft = torch.fft.fft(A, dim=1)

        # Step 2: Keep only the first rank_k frequency components
        matrix_fft_reduced = matrix_fft[:, :k]

        # Step 3: Take real part (or use both real and imag if needed)
        matrix_proj = matrix_fft_reduced.real
    
        return matrix_proj

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type),full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad,self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t().to(full_rank_grad.device.type), full_rank_grad) @ self.ortho_matrix[1].t().to(full_rank_grad.device.type)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0].to(low_rank_grad.device.type), low_rank_grad) @ self.ortho_matrix[1].to(low_rank_grad.device.type)


        return full_rank_grad * self.scale
    

    def get_orthogonal_matrix(self, weights, rank, type):
        """
        Use FFT-based projection to obtain a low-rank approximation "orthogonal" matrix.
        
        Args:
            weights: A module containing parameters, with a data attribute (tensor).
            rank (int): The target rank.
            type (str): 'left', 'right', or 'full', returning the left, right, or both matrices respectively.
        
        Returns:
            The corresponding low-rank approximation matrix or pair of matrices.
        """
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        # Compute the left-hand side approximation using FFT-based projection:
        # Get an approximation matrix of shape (m, rank)
        A_proj = fft_projection(matrix, k=rank)

        if type == 'left':
            result = A_proj
        elif type == 'right':
            # For the right-hand side matrix, we do the same for the matrix transpose.
            # We get a matrix of (n, rank), which is then transposed to (rank, n).
            A_proj_right = fft_projection(matrix.T, k=rank)
            result = A_proj_right.T
        elif type == 'full':
            A_left = A_proj
            A_proj_right = fft_projection(matrix.T, k=rank)
            A_right = A_proj_right.T
            result = [A_left, A_right]
        else:
            raise ValueError('type should be left, right or full')

        # If the original data is not float, convert it back to the original type and device
        def cast_back(x):
            if not float_data:
                return x.to(original_device).type(original_type)
            return x

        if isinstance(result, list):
            return [cast_back(result[0]), cast_back(result[1])]
        else:
            return cast_back(result)



import torch
from tensorly.decomposition import tucker
from tensorly import tenalg

# The GaLoreProjector class in Python implements a projection method using orthogonal matrix
# decomposition for low-rank approximation of gradients for general tensors of dimension >2.
# We use tensor decomposition using tensorly library: https://tensorly.org/stable/index.html
class GaLoreProjectorTensor:
    """
    A class that represents a projector for the GaLore algorithm.

    Args:
        rank (int): The rank of the projector.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        update_proj_gap (int, optional): The number of iterations between updating the orthogonal matrix. Defaults to 200.
        scale (float, optional): The scaling factor for the projected gradients. Defaults to 1.0.
    """

    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.transformed_low_rank = None

    def project(self, full_rank_grad, iter):
        """
        Projects the full-rank gradients onto the low-rank subspace.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            iter (int): The current iteration.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        if self.ortho_matrix is None and iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank)
        self.transformed_low_rank = self.transform(self.ortho_matrix, full_rank_grad)
        return self.transformed_low_rank

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.
        """
        full_rank_grad = self.inverse_transform(self.ortho_matrix, self.transformed_low_rank)
        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank_all):
        """
        Computes the orthogonal matrix using SVD decomposition.

        Args:
            weights (torch.Tensor): The weights to decompose.
            rank_all (int): The desired rank of the decomposition.

        Returns:
            tuple: A tuple containing the core and factors of the orthogonal matrix.
        """
        module_params = weights
        if module_params.data.dtype != torch.float:
            matrix = module_params.data.float()
        else:
            matrix = module_params.data
        tucker_tensor = tucker(matrix, rank=rank_all)
        return tucker_tensor

    def transform(self, tensor, x):
        """
        Transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, tensor, x):
        """
        Inverse transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The inverse transformed tensor.
        """
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors)


class FFT_GaLoreAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:

                if p.grad is None:
                    # num_has_no_grad += 1
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                # If 'dim' is not set in the group, default to 2
                if 'dim' not in group:
                    group['dim'] = 2

                # GaLore Projection
                if 'rank' in group:

                    if "projector" not in state:
                        # num_callMethod += 1
                        if group['dim'] <=2:
                            state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                        else:
                            state["projector"] = GaLoreProjectorTensor(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                # GaLore Projection Back
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


# ------------------------- Utility Functions -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Example: Function to generate prompt (can be modified according to actual situation)
def generate_SM(que: str) -> str:
    return (
        "You are a surgical AI agent assisting in pituitary surgery. Your job is to handle surgeons' queries efficiently by choosing appropriate text-promptable AI models and generating corresponding prompts.\n"
        "Available models: Segment-Video, Segment-MRI, Track-Instrument, Surgical-VQA, Overlaying.\n"
        "Question: {que}\n"
        "- Use ONE model if query focuses on a single, simple aspect:\n"
        "Example (single-model):\n"
        "Model: Segment-Video\nPrompt: Segment the sella in the video.\n"
        "- Use MULTIPLE models if query requires several types of information:\n"
        "Example (multi-model):\n"
        "Step1:\nModel: Segment-MRI\nPrompt: Segment the pituitary tumor from MRI.\n"
        "Step2:\nModel: Segment-Video\nPrompt: Segment the sella in the video.\n"
        "Now, follow the same format to answer the provided question—no extra text, labels, or formatting."
    ).format(que=que)


def extract_question(text):
    return text.strip() if pd.notna(text) else ""

def extract_answer(text):
    return text.strip() if pd.notna(text) else ""

# Process CSV data to build training and validation samples
def process_qa_samples(train_file, val_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    for df, name in [(train_df, 'Train.csv'), (val_df, 'Test.csv')]:
        if 'Input' not in df.columns or 'Label' not in df.columns:
            print(f"CSV file {name} is missing 'Input' or 'Label' column")
            return

    train_qa_samples = []
    for _, row in train_df.iterrows():
        question = extract_question(str(row['Input']))
        answer = extract_answer(str(row['Label']))
        if question and answer:
            question = generate_SM(question)
            train_qa_samples.append({"question": question, "answer": answer})

    valid_qa_samples = []
    for _, row in val_df.iterrows():
        question = extract_question(str(row['Input']))
        answer = extract_answer(str(row['Label']))
        if question and answer:
            question = generate_SM(question)
            valid_qa_samples.append({"question": question, "answer": answer})

    print("Train sample num:", len(train_qa_samples))
    print("Test sample num:", len(valid_qa_samples))
    if train_qa_samples:
        print("Example Train Sample:", train_qa_samples[0])
    if valid_qa_samples:
        print("Example Test Sample:", valid_qa_samples[0])
    
    return train_qa_samples, valid_qa_samples

# Data preprocessing
def preprocess_data(example, tokenizer, max_length=260):
    input_text = f"Query:\n{example['question']}\nResponse:\n{example['answer']}"
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length)
    labels = inputs["input_ids"].copy()
    question_length = len(tokenizer(f"Query:\n{example['question']}\nResponse:\n")["input_ids"]) - 1
    for i in range(len(labels)):
        if i < question_length or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100
    inputs["labels"] = labels
    return inputs

def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    return input_ids, attention_mask, labels

# Function to save the best model
def save_best_model(model, tokenizer, epoch, best_loss, current_loss, save_path):
    new_save_path = f"{save_path}_{epoch}"
    os.makedirs(new_save_path, exist_ok=True)
    model.save_pretrained(new_save_path)
    tokenizer.save_pretrained(new_save_path)
    print(f"Current model saved at epoch {epoch} with validation loss: {current_loss:.4f} in directory: {new_save_path}")
    if current_loss < best_loss:
        best_loss = current_loss
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Best model saved at epoch {epoch} with validation loss: {best_loss:.4f}")
    return best_loss

# ------------------------- Main Training Process -------------------------
args = parse_args()

# Set random seed
set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device string :", device)          # e.g. 'cuda:0' or 'cpu'
logging.warning("device string :" + str(device))  # e.g. 'cuda:0' or 'cpu'
print("device type   :", device.type)     # e.g. 'cuda' or 'cpu'
logging.warning("device type :" + str(device.type))  # e.g. 'cuda' or 'cpu'

if device.type == "cuda":
    print("device index  :", device.index)                     # which GPU ID
    print("device name   :", torch.cuda.get_device_name(0))    # GPU model
    print("CUDA version  :", torch.version.cuda)               # toolkit version
    logging.warning("device index :" + str(device.index))  # which GPU ID
    logging.warning("device name :" + str(torch.cuda.get_device_name(0)))  # GPU model
    logging.warning("CUDA version :" + str(torch.version.cuda))  # toolkit version

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",  # Normalized Float 4 (better than standard FP4)
#     bnb_4bit_use_double_quant=True,  # Uses secondary quantization for better precision
#     bnb_4bit_compute_dtype=torch.float16  # Keeps computation in FP16 for stability
# )

base_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    # quantization_config=bnb_config,
    device_map="auto"
)

base_model.config.pad_token_id = tokenizer.eos_token_id

model = base_model

galore_params = []
target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj"]
for module_name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    if not any(target_key in module_name for target_key in target_modules_list):
        continue
    print('enable GaLore for weights in module: ', module_name)
    galore_params.append(module.weight)
id_galore_params = [id(p) for p in galore_params]
# Other parameters are classified as regular_params
regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
# Define parameter groups, including additional GaLore configuration parameters
param_groups = [
    {'params': regular_params}, 
    {'params': galore_params, 
        'rank': args.rank, 
        'update_proj_gap': args.update_proj_gap, 
        'scale': args.galore_scale, 
        'proj_type': args.proj_type}
]

print("Number of elements in regular_params:", len(param_groups[0]['params']))
print("Number of elements in galore_params:", len(param_groups[1]['params']))
logging.warning(f"Number of elements in regular_params: {len(param_groups[0]['params'])}")
logging.warning(f"Number of elements in galore_params: {len(param_groups[1]['params'])}")
# Assume you already have the galore_params list and the corresponding id list
galore_ids = set(id(p) for p in galore_params)
for name, param in model.named_parameters():
    if id(param) in galore_ids:
        print(f"{name}: {param.requires_grad}")
        logging.warning(f"{name}: {param.requires_grad}")

# logging.warning("Parameter group configuration is as follows:")
# for i, group in enumerate(param_groups):
#     if i == 1:
#         logging.warning(f"Group {i}: {group}")
# if "rank" in param_groups[1]:
#     logging.warning(f"rank is {param_groups[1]['rank']}")
#     logging.warning("Fuck!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# else:
#     logging.warning("It's really not there, damn it...")
# logging.warning("Parameter group configuration ends")

trainable_params = param_groups

# Read and preprocess data
train_qa_samples, valid_qa_samples = process_qa_samples(args.train_file, args.val_file)
dataset_train = Dataset.from_list(train_qa_samples).map(lambda ex: preprocess_data(ex, tokenizer), remove_columns=["question", "answer"])
dataset_valid = Dataset.from_list(valid_qa_samples).map(lambda ex: preprocess_data(ex, tokenizer), remove_columns=["question", "answer"])

train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

optimizer = FFT_GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
print("DataLoaders and Optimizer configured!")

# 定义训练和验证函数
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = criterion(shift_logits, shift_labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

import time

def train(model, train_loader, valid_loader, optimizer, criterion, num_epochs, save_path):
    best_val_loss = float("inf")
    print("Start Training!")
    logging.warning("Start Training!")
    
    total_batches = len(train_loader)
    print(f"Total {total_batches} batches")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Record epoch start time
        model.train()
        total_train_loss = 0
        batch_times = []  # To store the time taken for each batch
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()  # Record batch start time
            
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = criterion(shift_logits, shift_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            batch_end_time = time.time()  # Record batch end time
            batch_time = batch_end_time - batch_start_time  # Calculate time taken for the current batch
            logging.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches} time: {batch_time:.4f}s")
            batch_times.append(batch_time)
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches} time: {batch_time:.4f}s")
            logging.warning(f"Loss: {loss.item()}")
        
        # Calculate average time per batch
        avg_batch_time = sum(batch_times) / len(batch_times)
        epoch_end_time = time.time()  # Record epoch end time
        epoch_time = epoch_end_time - epoch_start_time  # Calculate total time for the epoch
        
        avg_train_loss = total_train_loss / total_batches
        avg_val_loss = validate(model, valid_loader, criterion)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_val_loss:.4f}, Epoch time: {epoch_time:.2f}s, Average Batch time: {avg_batch_time:.4f}s")
        logging.warning(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
            f"Test Loss: {avg_val_loss:.4f}, Epoch time: {epoch_time:.2f}s, Average Batch time: {avg_batch_time:.4f}s, LR Description: {args.lr_description}"
        )
        best_val_loss = save_best_model(model, tokenizer, epoch + 1, best_val_loss, avg_val_loss, save_path)


# 开始训练
train(model, train_loader, valid_loader, optimizer, criterion, args.num_epochs, args.save_path)

#----------------------------------reference part----------------------------------------
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gc
import time
from transformers import pipeline
import evaluate
import pickle
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import requests
from torch import nn
from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaConfig, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union
from PIL import Image
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------- Utility Functions -------------------------
def generate_SM(que: str) -> str:
    return (
        "You are a surgical AI agent assisting in pituitary surgery. Your job is to handle surgeons' queries efficiently by choosing appropriate text-promptable AI models and generating corresponding prompts.\n"
        "Available models: Segment-Video, Segment-MRI, Track-Instrument, Surgical-VQA, Overlaying.\n"
        "Question: {que}\n"
        "- Use ONE model if query focuses on a single, simple aspect:\n"
        "Example (single-model):\n"
        "Model: Segment-Video\nPrompt: Segment the sella in the video.\n"
        "- Use MULTIPLE models if query requires several types of information:\n"
        "Example (multi-model):\n"
        "Step1:\nModel: Segment-MRI\nPrompt: Segment the pituitary tumor from MRI.\n"
        "Step2:\nModel: Segment-Video\nPrompt: Segment the sella in the video.\n"
        "Now, follow the same format to answer the provided question—no extra text, labels, or formatting."
    ).format(que=que)


def format_data(sample):
    system_message = generate_SM(sample[0])
    return [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": sample[1]}
    ]

def custom_collate_fn(sample):
    # Keep the format of samples returned by DataLoader unchanged
    return sample

def generate_answer(question, model, tokenizer):
    model.eval()
    input_text = f"Query:\n{question}\nResponse:\n"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0], skip_special_tokens=True).split("Response:\n")[-1].strip()
    return answer

def extract_prompt(entry):
    # Extract the content of all lines starting with "Prompt:" and join them with "|"
    prompts = [line[len("Prompt: "):].strip() for line in entry.split('\n') if line.startswith("Prompt:")]
    return "|".join(prompts) if prompts else ""

def group_by_sentence_position(all_prompts, num_sentences):
    grouped_sentences = [[] for _ in range(num_sentences)]
    for prompts in all_prompts:
        sentences = prompts.split("|")
        for i in range(num_sentences):
            if i < len(sentences):
                grouped_sentences[i].append(sentences[i])
            else:
                grouped_sentences[i].append("")
    return grouped_sentences

def compute_metrics(grouped_pred_prompts, grouped_ans_prompts):
    """
    Calculate ROUGE, BLEU, and METEOR metrics using the Hugging Face evaluate library.
    
    Parameters:
        grouped_pred_prompts: A list where each element is a group of predicted prompts (list of strings).
        grouped_ans_prompts: A list where each element is a group of true prompts (list of strings).
    
    Returns:
        rouge_results: A list of ROUGE metrics for each group.
        bleu_scores: A list of BLEU scores for each group.
        meteor_scores: A list of METEOR scores for each group.
    """
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    
    bleu_scores = []
    meteor_scores = []
    rouge_results = []
    
    for i in range(len(grouped_pred_prompts)):
        pred_group = grouped_pred_prompts[i]
        ans_group = grouped_ans_prompts[i]
        
        # ROUGE: Directly pass the list of predicted and reference texts
        rouge_result = rouge.compute(predictions=pred_group, references=ans_group)
        rouge_results.append(rouge_result)
        
        # BLEU: Reference texts need to be passed in a list of lists format
        bleu_result = bleu.compute(predictions=pred_group, references=[[ref] for ref in ans_group])
        bleu_scores.append({"bleu1": bleu_result["precisions"][0],
            "bleu2": bleu_result["precisions"][1],
            "bleu3": bleu_result["precisions"][2],
            "bleu4": bleu_result["precisions"][3]})
        
        # METEOR: Directly pass the list of predicted and reference texts
        meteor_result = meteor.compute(predictions=pred_group, references=ans_group)
        meteor_scores.append(meteor_result["meteor"])
    
    return rouge_results, bleu_scores, meteor_scores

def extract_model(text, model_names):
    # Use regular expressions to match model names from the list
    matches = re.findall(r'\b(?:' + '|'.join(map(re.escape, model_names)) + r')\b', text)
    return "|".join(matches) if matches else ""

def match_rate_per_Cat(pred_models_format, true_models_format):
    # Initialize counters
    first_model_match_count = 0
    second_model_match_count = 0
    third_model_match_count = 0
    total_count = len(true_models_format)

    # Iterate through both lists
    for pred, true in zip(pred_models_format, true_models_format):
        # Split model names
        pred_models = pred.split("|")
        true_models = true.split("|")
        while len(pred_models) < len(true_models):
            pred_models.append(" ")

        # check if the first model matches
        if  len(true_models) > 0 and pred_models[0] == true_models[0]:
            first_model_match_count += 1

        # check if the second model matches
        if len(true_models) > 1 and pred_models[1] == true_models[1]:
            second_model_match_count += 1
        
        if len(true_models) > 2 and pred_models[2] == true_models[2]:
            third_model_match_count += 1

    # Calculate matching percentage
    first_model_match_rate = (first_model_match_count / total_count * 100) if total_count > 0 else 0
    second_model_match_rate = (second_model_match_count / total_count * 100) if total_count > 0 else 0
    third_model_match_rate = (third_model_match_count / total_count * 100) if total_count > 0 else 0
    return first_model_match_rate, second_model_match_rate, third_model_match_rate

def f1_score_set(pred_list, true_list):
    pred_set = set(pred_list)
    true_set = set(true_list)
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(true_set) if true_set else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def evaluate_f1_by_selection_count(pred_models_format, true_models_format):
    """
    Calculate the F1 score for the agent when selecting 1, 2, and 3 models respectively.
    
    Parameters:
        pred_models_format: A list of strings, where multiple model names are separated by '|', representing the agent's predictions.
        true_models_format: A list of strings, where multiple model names are separated by '|', representing the true models.
    Returns:
        avg_one_model_f1: The average F1 score for all samples where the true number of models is 1.
        avg_two_model_f1: The average F1 score for all samples where the true number of models is 2.
        avg_three_model_f1: The average F1 score for all samples where the true number of models is 3.
    """
    one_model_scores = []
    two_model_scores = []
    three_model_scores = []
    
    for pred, true in zip(pred_models_format, true_models_format):
        # Split strings and remove extra spaces
        pred_models = [m.strip() for m in pred.split("|")]
        true_models = [m.strip() for m in true.split("|")]
        
        # If the number of predictions is insufficient, add empty strings (optional, adjust as needed)
        while len(pred_models) < len(true_models):
            pred_models.append("")
        
        # Calculate based on the number of true models
        if len(true_models) == 1:
            # For the case of one model, take the first one
            score = f1_score_set(pred_models[:1], true_models[:1])
            one_model_scores.append(score)
        elif len(true_models) == 2:
            # For the case of two models, take the first two
            score = f1_score_set(pred_models[:2], true_models[:2])
            two_model_scores.append(score)
        elif len(true_models) == 3:
            # For the case of three models, take the first three
            score = f1_score_set(pred_models[:3], true_models[:3])
            three_model_scores.append(score)
        else:
            # If the number of true models in the sample is not 1, 2, or 3, handle as needed (ignored here)
            continue

    avg_one_model_f1 = sum(one_model_scores) / len(one_model_scores) if one_model_scores else 0
    avg_two_model_f1 = sum(two_model_scores) / len(two_model_scores) if two_model_scores else 0
    avg_three_model_f1 = sum(three_model_scores) / len(three_model_scores) if three_model_scores else 0
    return avg_one_model_f1, avg_two_model_f1, avg_three_model_f1

class TextQuestionLabelDataset(Dataset):
    def __init__(self, input_file):
        self.data = pd.read_csv(input_file)
        self.questions = self.data['Input'].tolist()
        self.labels = self.data['Label'].tolist()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.labels[idx]

# ------------------------- Main Evaluation Process -------------------------
args = parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normalized Float 4 (better than standard FP4)
    bnb_4bit_use_double_quant=True,  # Uses secondary quantization for better precision
    bnb_4bit_compute_dtype=torch.float16  # Keeps computation in FP16 for stability
)
# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.best_model_path, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.best_model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Parse input file list
input_files = [f.strip() for f in args.input_files.split(",") if f.strip()]

for input_file in input_files:
    print(f"Processing file: {input_file}")
    logging.warning(f"Processing file: {input_file}")
    
    dataset = TextQuestionLabelDataset(input_file)
    test_dataset = [format_data(sample) for sample in dataset]
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    all_pred = []
    all_ans = []
    
    # Inference process
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            temp_pred = []
            temp_ans = []
            for sample in batch:
                output = generate_answer(sample[0]['content'], model, tokenizer)
                ans = sample[1]['content']
                temp_pred.append(output)
                temp_ans.append(ans)
            all_pred.extend(temp_pred)
            all_ans.extend(temp_ans)
    
    # Save prediction results
    base_filename = os.path.basename(input_file).replace(".csv", "")
    pred_output_file = os.path.join(args.output_dir, f"{base_filename}_pred.txt")
    with open(pred_output_file, "w") as f:
        for pred_text, ans_text in zip(all_pred, all_ans):
            f.write(f"pred: {pred_text}\n")
            f.write(f"ans: {ans_text}\n\n")
    print(f"Saved predictions to {pred_output_file}")
    
    # Prompt-based evaluation
    all_pred_prompts = all_pred
    all_ans_prompts = all_ans
    num_sentences = len(all_ans_prompts[0].split("|"))
    grouped_pred_prompts = group_by_sentence_position(all_pred_prompts, num_sentences)
    grouped_ans_prompts = group_by_sentence_position(all_ans_prompts, num_sentences)
    
    rouge_results, bleu_scores, meteor_scores = compute_metrics(grouped_pred_prompts, grouped_ans_prompts)
    
    # Model selection accuracy evaluation
    model_names = ["Segment-MRI", "Segment-Video", "Track-Instrument", "Surgical-VQA", "Overlaying"]
    pred_models = [extract_model(pred, model_names=model_names).strip() for pred in all_pred]
    true_models = [extract_model(ans, model_names=model_names).strip() for ans in all_ans]
    first_rate, second_rate, third_rate = match_rate_per_Cat(pred_models, true_models)
    # avg_two_f1, avg_three_f1 = evaluate_f1_by_selection_count(pred_models, true_models)
    true_models_list = true_models[0].split("|")
    model_num = len(true_models_list)
    
    avg_one_f1, avg_two_f1, avg_three_f1 = evaluate_f1_by_selection_count(pred_models, true_models)
    

    eval_output_file = os.path.join(args.output_dir, f"{base_filename}_evaluation.txt")
    with open(eval_output_file, "w") as f:
        f.write(f"Rouge Scores: {rouge_results}\n")
        f.write(f"BLEU Score: {bleu_scores}\n")
        f.write(f"METEOR Score: {meteor_scores}\n")
        if model_num > 0:
            f.write(f"Matching Accuracy of the 1st model: {first_rate:.2f}%\n")
            f.write(f"F1 score of current model: {avg_one_f1:.2f}\n")
        if model_num > 1:        
            f.write(f"Matching Accuracy of the 2nd model: {second_rate:.2f}%\n")
            f.write(f"F1 score for two models: {avg_two_f1:.2f}\n")
        if model_num > 2:
            f.write(f"Matching Accuracy of the 3rd model: {third_rate:.2f}%\n")
            f.write(f"F1 score for three models: {avg_three_f1:.2f}\n")
    print(f"Saved evaluation results to {eval_output_file}")
    logging.warning(f"Saved evaluation results to {eval_output_file}")
print("All files processed!")


# ========================= Code 1: Metric Extraction and Calculation =========================

def extract_evaluation_values(file_path):
    """
    Extract numerical values of various metrics from the evaluation.txt file:
      - rouge1, rougeL: Extracted from Rouge Scores
      - bleu1, bleu2, bleu3, bleu4: Extracted from BLEU Score
      - METEOR: Extracted from METEOR Score
      - F1: Extract F1 score related values
      - Matching Accuracy: Extract Matching Accuracy values
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract ROUGE metrics
    rouge1_matches = re.findall(r"'rouge1':\s*([0-9]*\.?[0-9]+)", content)
    rougeL_matches = re.findall(r"'rougeL':\s*([0-9]*\.?[0-9]+)", content)
    
    # Extract BLEU metrics
    bleu1_matches = re.findall(r"'bleu1':\s*([0-9]*\.?[0-9]+)", content)
    bleu2_matches = re.findall(r"'bleu2':\s*([0-9]*\.?[0-9]+)", content)
    bleu3_matches = re.findall(r"'bleu3':\s*([0-9]*\.?[0-9]+)", content)
    bleu4_matches = re.findall(r"'bleu4':\s*([0-9]*\.?[0-9]+)", content)
    
    # Extract METEOR metric
    meteor_matches = re.findall(r"'meteor':\s*([0-9]*\.?[0-9]+)", content)
    
    # Extract F1 metric (all types of F1 scores)
    file_name = os.path.basename(file_path)
    first_char = file_name.lstrip()[0] if file_name.lstrip() else ''
    if first_char == '2':
        f1_pattern = r"F1 score for two models:\s*([0-9]*\.?[0-9]+)"
    elif first_char == '3':
        f1_pattern = r"F1 score for three models:\s*([0-9]*\.?[0-9]+)"
    else:
        f1_pattern = r"F1 score of current model:\s*([0-9]*\.?[0-9]+)"
    f1_matches = re.findall(f1_pattern, content)
    
    matching_accuracy_matches = re.findall(
        r"Matching Accuracy of(?:\s+the)?\s+\d+(?:st|nd|rd|th)\s+model:\s*([0-9]*\.?[0-9]+)%",
        content
    )
    def to_float(lst):
        return [float(x) for x in lst]
    
    return {
        'rouge1': to_float(rouge1_matches),
        'rougeL': to_float(rougeL_matches),
        'bleu1': to_float(bleu1_matches),
        'bleu2': to_float(bleu2_matches),
        'bleu3': to_float(bleu3_matches),
        'bleu4': to_float(bleu4_matches),
        'METEOR': to_float(meteor_matches),
        'F1': to_float(f1_matches),
        'Matching_Accuracy': to_float(matching_accuracy_matches)
    }

def compute_average(values):
    """Calculate the average of a list, return 0 if the list is empty"""
    return sum(values) / len(values) if values else 0

def process_evaluation_files(output_dir):
    """
    Read all files ending with _evaluation.txt in output_dir,
    extract metric values, calculate averages, and save as average.txt and average.csv
    """
    all_values = {
        'rouge1': [],
        'rougeL': [],
        'bleu1': [],
        'bleu2': [],
        'bleu3': [],
        'bleu4': [],
        'METEOR': [],
        'F1': [],
        'Matching_Accuracy': []
    }
    
    evaluation_files = []
    # Iterate through all files in output_dir ending with _evaluation.txt
    for file in os.listdir(output_dir):
        if file.endswith("_evaluation.txt"):
            evaluation_files.append(file)
            file_path = os.path.join(output_dir, file)
            vals = extract_evaluation_values(file_path)
            for key in all_values:
                all_values[key].extend(vals[key])
    
    if not evaluation_files:
        print("No evaluation files found in {}".format(output_dir))
        return
    
    print("Found {} evaluation files: {}".format(len(evaluation_files), evaluation_files))
    
    # Calculate averages
    averages = {key: compute_average(all_values[key]) for key in all_values}
    
    # Generate average.txt file
    avg_txt_file = os.path.join(output_dir, "average.txt")
    with open(avg_txt_file, 'w', encoding='utf-8') as out:
        out.write("Average rouge1: {:.4f}\n".format(averages['rouge1']))
        out.write("Average rougeL: {:.4f}\n".format(averages['rougeL']))
        out.write("Average bleu1 Score: {:.4f}\n".format(averages['bleu1']))
        out.write("Average bleu2 Score: {:.4f}\n".format(averages['bleu2']))
        out.write("Average bleu3 Score: {:.4f}\n".format(averages['bleu3']))
        out.write("Average bleu4 Score: {:.4f}\n".format(averages['bleu4']))
        out.write("Average METEOR Score: {:.4f}\n".format(averages['METEOR']))
        out.write("Average F1 Score: {:.4f}\n".format(averages['F1']))
        out.write("Average Matching Accuracy: {:.4f}\n".format(averages['Matching_Accuracy']))
    print("Average results saved to: {}".format(avg_txt_file))
    
    # Generate average.csv file
    avg_csv_file = os.path.join(output_dir, "average.csv")
    with open(avg_csv_file, 'w', encoding='utf-8') as csv_out:
        # CSV header (column names)
        headers = ["Average bleu1 Score", "Average bleu2 Score", "Average bleu3 Score", 
                  "Average bleu4 Score", "Average rouge1", "Average rougeL",
                  "Average METEOR Score", "Average F1 Score", "Average Matching Accuracy"]
        
        # Multiply average values by 100 to convert to percentage values (rounded to two decimal places), Matching_Accuracy is not multiplied by 100 as it is already a percentage
        values = [
            averages['bleu1'] * 100,
            averages['bleu2'] * 100,
            averages['bleu3'] * 100,
            averages['bleu4'] * 100,
            averages['rouge1'] * 100,
            averages['rougeL'] * 100,
            averages['METEOR'] * 100,
            averages['F1'] * 100,
            averages['Matching_Accuracy']  # This is already a percentage, no need to multiply by 100
        ]
        
        # Write to CSV file, first write the header, then the data row
        csv_out.write(",".join(headers) + "\n")
        csv_out.write(",".join("{:.2f}".format(val) for val in values) + "\n")
    print("Average CSV file saved to: {}".format(avg_csv_file))
    
    return averages

# Automatically process evaluation files and generate average results
print("Processing evaluation files to generate average results...")
process_evaluation_files(args.output_dir)