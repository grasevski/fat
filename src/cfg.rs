//! Configuration hyperparameters.

/// Number of stacked GRU layers.
pub const LAYERS: usize = 1;

/// Number of hidden units in the GRU.
pub const FEATURES: usize = 128;

/// Number of consecutive observations.
pub const SEQ_LEN: usize = 120;

/// Length of an episode for reinforcement learning.
pub const STEPS: usize = 1;

/// Whether to carry observation between actions.
pub const STATEFUL: bool = true;

/// Whether GRU is bidirectional.
pub const BIDIRECTIONAL: bool = false;
