--[[
Implementation of
   Shim et al.,
   "SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks",
   NIPS 2017.

The module below replaces the nn.Linear module before the softmax module (nn.Softmax or nn.LogSoftmax)

--]]

local SVDLinear = torch.class('nn.SVDLinear', 'nn.Linear')

--[[
  self.weight: V x D
  self.weight_U: V x D
  self.weight_S: D
  self.weight_V: D x D

  To reconstruct weight: U * torch.diag(S) * V:t()
--]]
function SVDLinear:doSVD()
  assert(self.weight:nDimension() == 2)

  self.weight_U, self.weight_S, self.weight_V = torch.svd(self.weight)

  self.weight_B = self.weight_U * torch.diag(self.weight_S)   -- V x D


  assert(self.weight_U:nDimension() == 2)
  assert(self.weight_U:size(1) == self.weight:size(1))
  assert(self.weight_U:size(2) == self.weight:size(2))
end

function SVDLinear:setSVDParam(W, N)
  if W == nil or N == nil then
    self.svd_param_W = math.ceil(self.weight:size(2)/8)
    self.svd_param_N = math.ceil(self.weight:size(1)/16)
    return true
  end

  assert(type(W) == "number", "SVDLinear:setSVDParam(): W is not a number")
  assert(type(N) == "number", "SVDLinear:setSVDParam(): N is not a number")
  assert(W < self.weight:size(2), "SVDLinear:setSVDParam(): W is equal to or greater than input dimension")
  assert(N < self.weight:size(1), "SVDLinear:setSVDParam(): N is equal to or greater than output dimension")

  self.svd_param_W = W
  self.svd_param_N = N
end

function SVDLinear._updateFullView_lua(indices, z, B, h, bias)
  assert(indices:nDimension() == 1)
  assert(z:nDimension() == 1)
  assert(B:nDimension() == 2)
  assert(h:nDimension() == 1)
  assert(bias == nil or bias:nDimension() == 1)

  for i=1,indices:size(1) do
    local idx = indices[i]
    z[idx] = B[idx] * h
    if bias then
      z[idx] = z[idx] + bias[idx]
    end
  end
end

function SVDLinear._updateFullView_lua_batch(indices, z, B, h, bias)
  assert(indices:nDimension() == 2)
  assert(z:nDimension() == 2)
  assert(B:nDimension() == 2)
  assert(h:nDimension() == 2)
  assert(bias == nil or bias:nDimension() == 1)

  for b = 1,indices:size(2) do
    for i=1,indices:size(1) do
      local idx = indices[i][b]
      z[idx][b] = B[idx] * h[{{},b}]
      if bias then
        z[idx][b] = z[idx][b] + bias[idx]
      end
    end
  end
end

--[[
  input: B x D
  output: B x V

  self.weight: V x D
  self.bias: nil or V
--]]
function SVDLinear:updateOutput(input)
  assert(self.weight_U ~= nil)
  assert(self.weight_S ~= nil)
  assert(self.weight_V ~= nil)
  assert(self.svd_param_W ~= nil)
  assert(self.svd_param_N ~= nil)

  -- compute preview outputs with W dimensions
  local h_tilda, z_tilda
  z_tilda = self.output

  if input:nDimension() == 1 then
    h_tilda = self.weight_V:t() * input                 -- D
    z_tilda:resize(self.weight:size(1))
    z_tilda:mv(self.weight_B[{{},{1,self.svd_param_W}}],
               h_tilda:sub(1,self.svd_param_W))         -- V
    if self.bias then
      z_tilda:add(self.bias)
    end
  elseif input:nDimension() == 2 then
    h_tilda = self.weight_V:t() * input:t()             -- D x B
    z_tilda:resize(self.weight:size(1), input:size(1))
    z_tilda:mm(self.weight_B[{{},{1,self.svd_param_W}}],
               h_tilda:sub(1,self.svd_param_W))         -- V x B
    if self.bias then
      z_tilda:add(nn.utils.addSingletonDimension(self.bias,2):expand(self.bias:size(1), input:size(1)))
    end
  else
    assert(false, "nn.SVDLinear:updateOutput(): input should have either 1 or 2 dimensions")
  end

  -- select N words of largest preview outputs
  local _, Cn = torch.topk(z_tilda, self.svd_param_N, 1, true) -- retrieve top-k largest elements (C) and their indices (Cn)
  -- C: N (x B), Cn: N (x B)

  -- update selected words by full-view vector multiplication
  if z_tilda:nDimension() == 1 then
    self._updateFullView_lua(Cn, z_tilda, self.weight_B, h_tilda, self.bias)
--    self._updateFullView_ffi(Cn, z_tilda, self.weight_B, h_tilda, self.bias)
  elseif z_tilda:nDimension() == 2 then
    self._updateFullView_lua_batch(Cn, z_tilda, self.weight_B, h_tilda, self.bias)
--    self._updateFullView_ffi_batch(Cn, z_tilda, self.weight_B, h_tilda, self.bias)
    self.output = z_tilda:t()
  else
    assert(false, "nn.SVDLinear:updateOutput(): z_tilda should have either 1 or 2 dimensions")
  end

  -- (compute probability distribution using softmax): this is delegated to nn.LogSoftmax or nn.Softmax

  return self.output
end
