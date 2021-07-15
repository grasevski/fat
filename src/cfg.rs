//! Configuration hyperparameters.

/// Number of stacked GRU layers.
pub const LAYERS: usize = 2;

/// Number of hidden units in the GRU.
pub const FEATURES: usize = 256;

/// Number of consecutive observations.
pub const SEQ_LEN: usize = 3;

/// Length of an episode for reinforcement learning.
pub const STEPS: usize = 15;

/// Mini batch size.
pub const BATCHSIZE: usize = 3;
