//! Configuration hyperparameters.
use static_assertions::const_assert;

/// Number of stacked GRU layers.
pub const LAYERS: usize = 1;

/// Number of hidden units in the GRU.
pub const FEATURES: usize = 128;

/// Number of consecutive observations.
pub const SEQ_LEN: usize = 120;

/// Length of an episode for reinforcement learning.
pub const STEPS: usize = 1;

/// Mini batch size.
pub const BATCHSIZE: usize = STEPS;

const_assert!(STEPS % BATCHSIZE == 0);
