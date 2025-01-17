#pragma once

#include <vector>

namespace ctcdecode {

/* Struct for the beam search output, containing the tokens based on the
 * vocabulary indices, and the timesteps for each token in the beam search
 * output
 */
struct Output {
  std::vector<int> tokens, timesteps;
};

} // namespace ctcdecode
