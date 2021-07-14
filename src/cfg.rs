//! Configuration hyperparameters.

/// Number of stacked GRU layers.
pub const LAYERS: usize = 1;

/// Number of hidden units in the GRU.
pub const FEATURES: usize = 1;

/// Number of consecutive observations.
pub const SEQ_LEN: usize = 1;

/// Length of an episode for reinforcement learning.
pub const STEPS: usize = 1;

/// Mini batch size.
pub const BATCHSIZE: usize = 1;
