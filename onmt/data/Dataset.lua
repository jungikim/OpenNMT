--[[ Data management and batch creation. Handles data created by `preprocess.lua`. ]]
local Dataset = torch.class("Dataset")


function Dataset.openDB_RO(path)
  local db = lmdb.env {Path = path}
  db:open()
  local txn = db:txn(true)
  return db, txn
end

function Dataset.closeDB(db, txn)
  txn:commit()
  db:close()
end

--[[ Initialize a data object given aligned tables of IntTensors `srcData`
  and `tgtData`.
--]]
function Dataset:__init(srcData, tgtData)

  self.src = srcData.words or srcData.vectors
  self.srcFeatures = srcData.features
  self.constraints = srcData.constraints

  if type(self.src) ~= 'table' and type(self.src) == 'string' then
    require('lmdb')
    self.src_db, self.src_db_txn = self.openDB_RO(self.src)
    self.srcFeatures_db, self.srcFeatures_db_txn = self.openDB_RO(self.srcFeatures)
  end

  if tgtData ~= nil then
    self.tgt = tgtData.words or tgtData.vectors
    self.tgtFeatures = tgtData.features

    if type(self.tgt) ~= 'table' and type(self.tgt) == 'string' then
      self.tgt_db, self.tgt_db_txn = self.openDB_RO(self.tgt)
      self.tgtFeatures_db, self.tgtFeatures_db_txn = self.openDB_RO(self.tgtFeatures)
    end
  end
end

function Dataset:getSrc(i)
  if self.src_db then
    return self.src_db_txn:get(i)
  end
  return self.src[i]
end

function Dataset:getSrcFeature(i)
  if self.srcFeatures_db and self.srcFeatures_db:stat()['entries'] >= i then
    return self.srcFeatures_db_txn:get(i)
  end
  return self.srcFeatures[i]
end

function Dataset:getTgt(i)
  if self.tgt_db then
    return self.tgt_db_txn:get(i)
  end
  return self.tgt[i]
end

function Dataset:getTgtFeature(i)
  if self.tgtFeatures_db and self.tgtFeatures_db:stat()['entries'] >= i then
    return self.tgtFeatures_db_txn:get(i)
  end
  return self.tgtFeatures[i]
end

function Dataset:getSrcSize()
  if self.src_db then
    return self.src_db:stat()['entries']
  end
  return #self.src
end

--[[ Setup up the training data to respect `maxBatchSize`.
     If uneven_batches - then build up batches with different lengths ]]
function Dataset:setBatchSize(maxBatchSize, uneven_batches)

  self.batchRange = {}
  self.maxSourceLength = 0
  self.maxTargetLength = 0

  local batchesCapacity = 0
  local batchesOccupation = 0

  -- Prepares batches in terms of range within self.src and self.tgt.
  local offset = 0
  local batchSize = 1
  local maxSourceLength = 0
  local targetLength = 0

  for i = 1, self:getSrcSize() do
    -- Set up the offsets to make same source size batches of the
    -- correct size.
    local sourceLength = self:getSrc(i):size(1)
    if batchSize == maxBatchSize or i == 1 or
        (not(uneven_batches) and sourceLength ~= maxSourceLength) then
      if i > 1 then
        batchesCapacity = batchesCapacity + batchSize * maxSourceLength
        table.insert(self.batchRange, { ["begin"] = offset, ["end"] = i - 1 })
      end

      offset = i
      batchSize = 1
      targetLength = 0
      maxSourceLength = 0
    else
      batchSize = batchSize + 1
    end
    batchesOccupation = batchesOccupation + sourceLength
    maxSourceLength = math.max(maxSourceLength, sourceLength)

    self.maxSourceLength = math.max(self.maxSourceLength, sourceLength)

    if self.tgt ~= nil then
      -- Target contains <s> and </s>.
      local targetSeqLength = self:getTgt(i):size(1) - 1
      targetLength = math.max(targetLength, targetSeqLength)
      self.maxTargetLength = math.max(self.maxTargetLength, targetSeqLength)
    end
  end

  -- Catch last batch.
  batchesCapacity = batchesCapacity + batchSize * maxSourceLength
  table.insert(self.batchRange, { ["begin"] = offset, ["end"] = self:getSrcSize() })
  return #self.batchRange, batchesOccupation/batchesCapacity
end

--[[ Return number of batches. ]]
function Dataset:batchCount()
  if self.batchRange == nil then
    if self:getSrcSize() > 0 then
      return 1
    else
      return 0
    end
  end
  return #self.batchRange
end

function Dataset:instanceCount()
  return self:getSrcSize()
end

--[[ Get `Batch` number `idx`. If nil make a batch of all the data. ]]
function Dataset:getBatch(idx)
  if self:getSrcSize() == 0 then
    return nil
  end

  local rangeStart = (idx and self.batchRange) and self.batchRange[idx]["begin"] or 1
  local rangeEnd = (idx and self.batchRange) and self.batchRange[idx]["end"] or self:getSrcSize()

  local src = {}
  local tgt

  if self.tgt ~= nil then
    tgt = {}
  end

  local srcFeatures = {}
  local tgtFeatures = {}

  local constraints = {}

  for i = rangeStart, rangeEnd do
    table.insert(src, self:getSrc(i))

    if self:getSrcFeature(i) then
      table.insert(srcFeatures, self:getSrcFeature(i))
    end

    if self.tgt ~= nil then
      table.insert(tgt, self:getTgt(i))

      if self:getTgtFeature(i) then
        table.insert(tgtFeatures, self:getTgtFeature(i))
      end
    end

    if self.constraints and self.constraints[i] then
      table.insert(constraints, self.constraints[i])
    end
  end

  return onmt.data.Batch.new(src, srcFeatures, tgt, tgtFeatures, constraints)
end

return Dataset
