#include "attention.h"

namespace mxflashattn {

int64_t workspace_bytes(int64_t batch, int64_t q_len, int64_t kv_len, int64_t head_dim) {
  TORCH_CHECK(batch >= 0 && q_len >= 0 && kv_len >= 0 && head_dim >= 0, "workspace dimensions must be non-negative");
  // The ATen bring-up backend owns its temporary tensors through PyTorch.
  return 0;
}

}  // namespace mxflashattn
