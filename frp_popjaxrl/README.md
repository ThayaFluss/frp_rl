# FRP PopJaxRL

Core implementation of Free Random Projection (FRP) for meta-reinforcement learning using JAX. This module provides the main FRP algorithm, meta-environment framework, and RL algorithms with in-context learning capabilities.

## Project Structure

- `frp/orthogonal.py`: Core FRP algorithm - orthogonal matrix generation and word composition
- `envs/meta_environment.py`: Meta-environment framework for FRP transformations
- `algorithms/`: PPO implementations with S5/GRU architectures for in-context learning
- `run_meta_popgym.py`: Main experiment runner

## Example Usage

To run a meta-learning experiment with S5 architecture on the NoisyStatelessCartPole environment (POMDP version of cartpole):

```bash
python3 run_meta_popgym.py --arch s5 --depth 2 --dim 32 --env cartpole --max_depth 8 --seed 42
```

Key parameters:
- `--arch`: Neural network architecture (s5 or gru)
- `--depth`: Depth of matrix composition in FRP
- `--dim`: Output dimension of the FRP
- `--env`: Environment name (e.g., cartpole, minesweeper)
- `--max_depth`: Maximum depth (controls number of matrices in FRP)
- `--seed`: Random seed for reproducibility

## Evaluating Trained Models

After training, use the appropriate evaluation script based on the training mode:

### For Separated Mode (Recommended)

```bash
# Evaluate a separated mode checkpoint
python run_eval_separated.py --checkpoint exp/20260106_122337/model_31_iter.pkl \
    --eval_num_trials 32 \
    --num_episodes 10 \
    --eval_method tiling
```

### For Legacy/Lazy Modes

```bash
# Legacy mode evaluation
python run_eval.py --checkpoint exp/model.pkl --mode legacy \
    --eval_num_trials 32 --num_episodes 10

# Lazy mode evaluation
python run_eval.py --checkpoint exp/model.pkl --mode lazy \
    --eval_num_trials 32 --num_episodes 10
```

**Evaluation Methods:**
- `tiling` (recommended): Tiles observations to fill meta_dim
- `padding`: Pads with zeros
- `identity`: No FRP transformation (ablation)

**Outputs:**
- Console statistics per trial
- PNG plots (returns, steps, success rates)
- CSV data for analysis
- Optional WandB logging

See `.claude/CLAUDE.md` for detailed evaluation instructions.

# How `frp/orthogonal.py` is Used in `run_meta_popgym.py`

## Overview

The `frp/orthogonal.py` module is a core component of the Free Random Projection (FRP) system used in meta-learning environments. This document summarizes how this module is utilized when `run_meta_popgym.py` is executed.

## Key Components and Flow

### 1. Initialization in `run_meta_popgym.py`

When `run_meta_popgym.py` is executed, it:

1. Parses command-line arguments, including FRP-specific parameters:
   - `--dim`: Output dimension of the FRP (default: 64)
   - `--depth`: Depth of matrix composition in FRP (default: 2)
   - `--max_depth`: Maximum depth (controls number of matrices in FRP) (default: 8)
   - `--with_adjoint`: Whether to use adjoint of orthogonal matrices (default: 0/False)

2. These parameters are collected into a `meta_kwargs` dictionary:
   ```python
   meta_kwargs = {
       "meta_depth": args.depth,
       "meta_max_depth": args.max_depth,
       "meta_dim": args.dim,
       "meta_with_adjoint": (args.with_adjoint==1),
   }
   ```

3. The script then calls `create_meta_environment()` with these parameters to create both training and evaluation environments.

### 2. Meta-Environment Creation

The `create_meta_environment()` function:

1. Selects the appropriate environment class based on the environment name (e.g., "cartpole", "minesweeper")
2. Instantiates a `MetaEnvironment` class with the selected environment and the provided `meta_kwargs`

### 3. MetaEnvironment Initialization

When a `MetaEnvironment` is instantiated:

1. It extracts FRP parameters from `meta_kwargs`:
   - `meta_dim`: Output dimension of projection
   - `meta_depth`: Depth of orthogonal matrix composition
   - `meta_max_depth`: Maximum depth (controls number of words)
   - `meta_with_adjoint`: Whether to use adjoint matrices

2. It calls `_initialize_meta_augmentation()` which uses functions from `frp/orthogonal.py`:

   ```python
   def _initialize_meta_augmentation(self):
       matrices = create_orthogonal_matrices(
           self.rng, 
           self.meta_depth, 
           size=self.meta_dim, 
           max_depth=self.meta_max_depth, 
           with_adjoint=self.meta_with_adjoint
       )
       self.words = create_words(
           matrices, 
           self.meta_depth, 
           out_size=self.meta_dim, 
           max_depth=self.meta_max_depth
       )
       # Exclude identity matrices before cutoff (needless if with_adjoint=False)
       self.exclude = detect_identity_matrices(self.words)
       # ... additional processing ...
   ```

### 4. Key Functions from `frp/orthogonal.py`

The following functions from `orthogonal.py` are crucial to the meta-environment:

#### a. `create_orthogonal_matrices()`
- Creates a set of random orthogonal matrices using QR decomposition
- Parameters:
  - `key`: JAX random key
  - `depth`: Depth of matrix composition
  - `size`: Dimension of matrices
  - `max_depth`: Maximum depth (controls number of matrices)
  - `with_adjoint`: Whether to include transpose of matrices

#### b. `create_words()`
- Generates word matrices by composing orthogonal matrices
- Creates 2^max_depth different words
- Each word is created by multiplying a sequence of orthogonal matrices

#### c. `detect_identity_matrices()`
- Identifies which word matrices are identity matrices
- These are excluded from random selection to ensure non-trivial transformations

#### d. `get_weight_matrix()`
- Extracts a specific weight matrix for a given environment index
- Used during environment steps to transform observations
- Optimized for JAX's JIT compilation

### 5. Periodic Word Generation During Training

The `orthogonal.py` functions are not only used during environment initialization but are also called periodically during training. This is controlled by the `reset_words` parameter in `run_meta_popgym.py`:

```python
parser.add_argument("--reset_words", type=int, default=1, help="reset words per epoch")
```

In the PPO implementation, word regeneration happens at a specific level in the training hierarchy:

1. **Environment Step**: A single interaction with the environment (action → observation → reward)
2. **Training Batch**: Collection of multiple environment steps (NUM_ENVS × NUM_STEPS interactions)
3. **Update Step**: Processing of a full batch to update the neural network weights
4. **Training Loop**: Multiple update steps until reaching TOTAL_TIMESTEPS

The word regeneration occurs at the beginning of each **update step** in the training loop, before collecting a new batch of environment interactions:

```python
# Inside the _update_step function in ppo_gru_in_context.py
if config.get("RESET_WORDS"):
    if config["RESET_WORDS"]:
        # Update words after each update step (before collecting new trajectories)
        rng, _rng = jax.random.split(rng)
        words = _create_words(_rng)
        
        # Update environment words
        env.words = words
        env_state.env_state.replace(
            obs_words=words)
```

The `_create_words` function regenerates the orthogonal matrices and words:

```python
def _create_words(key):
    matrices = create_orthogonal_matrices(
        key,
        config["ENV"].meta_depth,
        size=config["ENV"].meta_dim,
        max_depth=config["ENV"].meta_max_depth,
        with_adjoint=config["ENV"].meta_with_adjoint
    )
    words = create_words(
        matrices,
        config["ENV"].meta_depth,
        out_size=config["ENV"].meta_dim,
        max_depth=config["ENV"].meta_max_depth
    )
    # ... additional processing ...
    return words
```

This periodic regeneration ensures that:
1. Each update step uses a consistent set of words for all environment steps within that batch
2. Different update steps use different words, exposing the agent to a diverse set of transformations
3. The agent must learn to adapt to new observation transformations, improving meta-learning capabilities

### 6. Usage During Environment Steps

During environment execution:

1. In `reset_env()`, a random word index is selected using `random_choice()`:
   ```python
   env_index = random_choice(index_key, 
                          total_words=self.total_words, 
                          exclude=self.exclude).astype(jnp.int32)
   ```

2. The selected word is used to transform observations:
   ```python
   weight = get_weight_matrix(self.words, env_index, self.input_dim, self.aug_output_dim)
   augmented_obs = (env_obs[None,:] @ weight)[0]
   ```

3. Similarly, in `step_env()`, the same word is used consistently throughout an episode:
   ```python
   weight = get_weight_matrix(state.obs_words, state.env_index, self.input_dim, self.aug_output_dim)
   env_obs = (env_obs[None,:] @ weight)[0]
   ```

### 7. Evaluation Methods

The system supports different evaluation methods:

1. **Identity**: Uses identity matrix (no transformation)
2. **Padding**: Uses a padding matrix that preserves original dimensions
3. **Tiling**: Uses a periodic weight matrix

These are controlled by the `--eval_method` parameter in `run_meta_popgym.py`.

## Summary

The `frp/orthogonal.py` module provides the mathematical foundation for Free Random Projection in meta-learning environments. It generates orthogonal transformation matrices that are used to create diverse task variations while preserving inner products of the input space. These transformations are applied consistently within episodes but vary between episodes, enabling the agent to learn meta-learning capabilities.

The key innovation is using orthogonal matrices to transform observations in a way that preserves their inner products while creating diverse task variations. This approach allows for effective meta-learning across a range of environments.


## Key Implementation Details

### FRP Algorithm Flow
1. **Matrix Generation**: Creates 2^max_depth orthogonal matrices via QR decomposition
2. **Word Composition**: Generates transformation matrices by composing orthogonal matrices
3. **Episode Selection**: Randomly selects a transformation at episode start
4. **Consistent Application**: Applies the same transformation throughout an episode
5. **Periodic Regeneration**: Creates new transformations between training batches for diversity

### In-Context Learning
During evaluation, agents adapt to new orthogonal transformations by **updating only their hidden states, not parameters**:

- **No Parameter Updates**: Model weights remain frozen during testing
- **Hidden State Adaptation**: Only recurrent hidden states (GRU/S5) update as the agent observes the transformed environment
- **Online Learning**: Adaptation happens in real-time through hidden state dynamics within each episode
