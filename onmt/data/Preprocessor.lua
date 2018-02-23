--[[ Data Preparation functions. ]]

local function vecToTensor(vec)
  local t = torch.Tensor(#vec)
  for i = 1, #vec do
    t[i] = vec[i]
  end
  return t
end

local Preprocessor = torch.class('Preprocessor')
local paths = require 'paths'
local tokenizer = require('tools.utils.tokenizer')
local lmdb = nil

local tds
local threads

local preprocess_batchsize = 10000

local commonOptions = {
  {
    '-features_vocabs_prefix', '',
    [[Path prefix to existing features vocabularies.]]
  },
  {
    '-time_shift_feature', true,
    [[Time shift features on the decoder side.]]
  },
  {
    '-keep_frequency', false,
    [[Keep frequency of words in dictionary.]]
  },
  {
    '-gsample', 0,
    [[If not zero, extract a new sample from the corpus. In training mode, file sampling is done at each epoch. Values between 0 and 1 indicate ratio,
      values higher than 1 indicate data size]],
    {
      valid = onmt.utils.ExtendedCmdLine.isFloat(0)
    }
  },
  {
    '-gsample_dist', '',
    [[Configuration file with data class distribution to use for sampling training corpus. If not set, sampling is uniform.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      depends = function(opt) return opt.gsample_dist == '' or opt.gsample > 0, "option `gsample_dist` requires `gsample`" end
    }
  },
  {
    '-sort', true,
    [[If set, sort the sequences by size to build batches without source padding.]]
  },
  {
    '-shuffle', true,
    [[If set, shuffle the data (prior sorting).]]
  },
  {
    '-idx_files', false,
    [[If set, source and target files are 'key value' with key match between source and target.]]
  },
  {
    '-report_progress_every', 100000,
    [[Report status every this many sentences.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-preprocess_pthreads', 4,
    [[Number of parallel threads for preprocessing.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-lmdb', false,
    [[Store data instances in LMDB]],
    {
      depends = function(opt)
                  return opt.lmdb == false or (opt.sort == false and opt.shuffle == false),
                         "option `lmdb` requires `sort`=false and `shuffle`=false"
                end
    }
  }
}

function Preprocessor.getDataList(dataType)
  local datalist
  if dataType == 'bitext' or dataType == 'seq2seq' then
    datalist = { {name="source",short="src",hasVocab=true,suffix=".src"} , {name="target",short="tgt",hasVocab=true,suffix=".tgt"} }
  elseif dataType == 'monotext' then
    datalist = { {hasVocab=true,suffix=".tok"} }
  else
    datalist = { {name="source",short="src",hasVocab=false,suffix=".src"} , {name="target",short="tgt",hasVocab=true,suffix=".tgt"} }
  end
  return datalist
end

-- utility functions
local function prefix(data)
  if data.short then return data.short.."_" end
  return ""
end
local function suffix(data)
  if data.short then return "_"..data.short end
  return ""
end
local function nameWithSpace(data)
  if data.name then return " "..data.name end
  return ""
end

--[[
  Generic function to generate options for the different dataTypes
]]
local function declareDataOptions(dataType)
  local datalist = Preprocessor.getDataList(dataType)
  local options = {}

  table.insert(options,
    {
      '-train_dir', '',
      [[Path to training files directory.]],
      {
        valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
      }
    })
  for i = 1, #datalist do
    table.insert(options,
      {
        '-train'..suffix(datalist[i]), '',
        "Path to the training"..nameWithSpace(datalist[i]).." data.",
        {
          valid=onmt.utils.ExtendedCmdLine.fileNullOrExists
        }
      })
  end

  for i = 1, #datalist do
    table.insert(options,
      {
        '-valid'..suffix(datalist[i]), '',
        "Path to the validation"..nameWithSpace(datalist[i]).." data.",
        {
          valid=onmt.utils.ExtendedCmdLine.fileNullOrExists
        }
      })
  end
  for i = 1, #datalist do
    if datalist[i].hasVocab then
      table.insert(options,
        {
          '-'..prefix(datalist[i])..'vocab', '',
          "Path to an existing"..nameWithSpace(datalist[i]).." vocabulary.",
          {
            valid=onmt.utils.ExtendedCmdLine.fileNullOrExists
          }
        })
      table.insert(options,
        {
          '-'..prefix(datalist[i])..'suffix', datalist[i].suffix,
          "Suffix for"..nameWithSpace(datalist[i]).." files in train/valid directories."
        })
      table.insert(options,
        {
          '-'..prefix(datalist[i])..'vocab_size', { 50000 },
          "List of"..nameWithSpace(datalist[i])..[[ vocabularies size: `word[ feat1[ feat2[ ...] ] ]`.
            If = 0, vocabularies are not pruned.]]
        })
      table.insert(options,
        {
          '-'..prefix(datalist[i])..'words_min_frequency', { 0 },
          "List of"..nameWithSpace(datalist[i])..[[ words min frequency: `word[ feat1[ feat2[ ...] ] ]`.
            If = 0, vocabularies are pruned by size.]]
        })
    end
  end
  for i = 1, #datalist do
    table.insert(options,
      {
        '-'..prefix(datalist[i])..'seq_length', 50,
        "Maximum"..nameWithSpace(datalist[i])..[[ sequence length.]],
        {
          valid = onmt.utils.ExtendedCmdLine.isInt(1)
        }
      })
  end

  if dataType ~= 'monotext' then
    table.insert(options,
      {
        '-check_plength', false,
        [[Check source and target have same length (for seq tagging).]]
      })
  end

  return options
end

function Preprocessor.declareOpts(cmd, dataType)
  dataType = dataType or 'bitext'
  local options = declareDataOptions(dataType)
  for _, v in ipairs(commonOptions) do
    table.insert(options, v)
  end
  cmd:setCmdLineOptions(options, 'Data')

  -- prepare tokenization option
  local topts = tokenizer.getOpts()
  for i, v in ipairs(topts) do
    if v[1] == '-mode' then
      topts[i] = {
          '-mode', 'space',
          [[Define how aggressive should the tokenization be. `space` is space-tokenization.]],
            {
              enum = {'conservative', 'aggressive', 'space'}
            }
          }
    end
  end

  cmd:setCmdLineOptions(topts, "Tokenizer")
end

function Preprocessor.expandOpts(cmd, dataType)
  local torenameOpts = {};
  local current_block;
  local pref = "{src,tgt}_"
  if dataType == "monotext" then pref = "" end
  if dataType == "feattext" then pref = "tgt_" end
  if dataType == "audiotext" then pref = "tgt_" end
  for i, v in ipairs(cmd.helplines) do
    if type(v) == "string" then
      local p = v:find(" options")
      if p then
        current_block = v:sub(1,p-1);
        if current_block == "MPreprocessing" or current_block == "Tokenizer" then
          cmd.helplines[i] = cmd.helplines[i]
        end
      end
    else
      if current_block == "MPreprocessing" or current_block == "Tokenizer" then
        torenameOpts[v.key] = current_block:sub(1,3):lower()
        v.key="-"..current_block:sub(1,3):lower().."_"..pref..v.key:sub(2)
      end
    end
  end
  local newOpts = {}
  for k, v in pairs(cmd.options) do
    if torenameOpts[k] then
      cmd.options[k] = nil
      if dataType == 'monotext' then
        local ksrc = '-'..torenameOpts[k]..'_'..k:sub(2)
        newOpts[ksrc] = onmt.utils.Table.deepCopy(v)
      elseif dataType == 'bitext' then
        local ksrc = '-'..torenameOpts[k]..'_src_'..k:sub(2)
        newOpts[ksrc] = onmt.utils.Table.deepCopy(v)
      end
      if dataType ~= 'monotext' then
        local ktgt = '-'..torenameOpts[k]..'_tgt_'..k:sub(2)
        newOpts[ktgt] = onmt.utils.Table.deepCopy(v)
      end
    end
  end
  for k, v in pairs(newOpts) do
    cmd.options[k] = v
  end
end

local function ruleMatch(s, rule)
  if rule == '*' then return true end
  local pat = onmt.utils.String.split(rule, ",")
  for _, r in ipairs(pat) do
    if string.match(s, r) then return true end
  end
end

function Preprocessor:parseDirectory(args, datalist, dist_rules, keep_rules, type)
  local dir = args[type.."_dir"]
  assert(dir ~= '', 'missing \''..type..'_dir\' parameter')
  _G.logger:info('Parsing '..type..' data from directory \''..dir..'\':')
  local firstSuffix = args[prefix(datalist[1])..'suffix']

  local totalCount = 0
  local totalError = 0
  local list_files = {}

  for candf in paths.iterfiles(dir) do
    if firstSuffix == '' or candf:sub(-firstSuffix:len()) == firstSuffix then
      self:poolAddJob(
        function(f)
          local flist = {}
          local errors = {}
          local fprefix = f:sub(1, -firstSuffix:len()-1)
          table.insert(flist, _G.paths.concat(dir,f))
          local error = 0
          local countLines = onmt.utils.FileReader.countLines(flist[1], args.idx_files)
          for i = 2, #datalist do
            local tfile = _G.paths.concat(dir,fprefix..args[prefix(datalist[i])..'suffix'])
            table.insert(flist, tfile)
            if not _G.path.exists(tfile) or onmt.utils.FileReader.countLines(tfile, args.idx_files) ~= countLines then
              table.insert(errors, '* ['.._G.__threadid..'] invalid file - '..tfile..' - not aligned with '..f)
              error = error + 1
            end
          end
          if error == 0 then
            local fdesc = { countLines, flist }
            fdesc.fname = fprefix
            fdesc.weight = 0
            fdesc.options = {}
            return _G.__threadid, 0, fdesc
          else
            return _G.__threadid, error, errors
          end
        end,
        function(threadid, error, fdesc)
          if error > 0 then
            totalError = totalError + error
            for _, m in ipairs(fdesc) do
              _G.logger:error(m)
            end
          else
            _G.logger:info(' * ['..threadid..'] Reading files \''..fdesc.fname..'\' - '..fdesc[1]..' sentences')
            table.insert(list_files, fdesc)
            totalCount = totalCount + fdesc[1]
          end
        end,
        candf)
    end
  end

  self:poolSynchronize()

  if totalError > 0 then
    _G.logger:error('Errors in training directory - fix them first')
    os.exit(0)
  end
  if totalCount == 0 then
    _G.logger:error('No '..type..' data found in directory \''..dir..'\'')
    os.exit(0)
  end
  _G.logger:info(totalCount..' sentences, in '..#list_files..' files, in '..type..' directory')
  _G.logger:info('')

  local keepCount = 0

  if #keep_rules > 0 then
    _G.logger:info('Matching files with keep rules:')
    for i = 1, #list_files do
      if list_files[i].weight == 0 then
        for rule_idx = 1, #keep_rules do
          if ruleMatch(list_files[i].fname, keep_rules[rule_idx][1]) then
            keepCount = keepCount + list_files[i][1]
            list_files[i].rule_idx = rule_idx
            list_files[i].weight = math.huge
            list_files[i].options = keep_rules[rule_idx][3]
            _G.logger:info(" * file '%s' is covered by the keep rule %d",
                           list_files[i].fname, list_files[i].rule_idx or 0)
            break
          end
        end
      end
    end
    _G.logger:info('')

    -- Files matched with keep rules are not part of the global sampling.
    totalCount = totalCount - keepCount
  end

  if #dist_rules > 0 then
    _G.logger:info('Matching files with sample rules:')
    local weight_norm = 0
    local weight_rule = {}

    for i = 1, #list_files do
      if list_files[i].weight == 0 then
        for rule_idx = 1, #dist_rules do
          if ruleMatch(list_files[i].fname, dist_rules[rule_idx][1]) then
            list_files[i].rule_idx = rule_idx
            if not weight_rule[rule_idx] then
              weight_norm = weight_norm + dist_rules[rule_idx][2]
              weight_rule[rule_idx] = 0
            end
            weight_rule[rule_idx] = weight_rule[rule_idx] + list_files[i][1]
            break
          end
        end
      end
    end

    local sum_weight = 0
    for i = 1, #list_files do
      if list_files[i].rule_idx and list_files[i].weight ~= math.huge then
        local rule_idx = list_files[i].rule_idx
        list_files[i].weight = dist_rules[rule_idx][2] / weight_norm * list_files[i][1] / weight_rule[rule_idx]
        sum_weight = sum_weight + list_files[i].weight
        list_files[i].options = dist_rules[rule_idx][3]
      end
    end

    for i = 1, #list_files do
      if list_files[i].weight ~= math.huge then
        list_files[i].weight = list_files[i].weight / sum_weight
        if list_files[i].weight > 0 then
          _G.logger:info(" * file '%s' is covered by the sampling rule %d - uniform weight: %.4f, distribution weight: %.4f",
                         list_files[i].fname,
                         list_files[i].rule_idx or 0,
                         100 * list_files[i][1] / totalCount,
                         100 * list_files[i].weight)
        end
      end
    end

    _G.logger:info('')
  else
    for i = 1, #list_files do
      list_files[i].weight = list_files[i][1] / totalCount
    end
  end

  for i = 1, #list_files do
    if list_files[i].weight == 0 then
      _G.logger:warning(" * file '%s' is not covered by any rules and will not be used",
                        list_files[i].fname)
    end
  end

  return totalCount, keepCount, list_files
end

-- helper functions for threading
function Preprocessor:poolAddJob(f, r, ...)
  if self.pool then
    self.pool:addjob(f, r, ...)
  else
    _G.__threadid = '-'
    _G.tds = tds
    r(f(...))
  end
end

function Preprocessor:poolSynchronize()
  if self.pool then
    self.pool:synchronize()
  end
end

-- initialization of threads and optTok
local function init_thread(id, args, optTok, optMPr)
  _G.paths = require 'paths'
  _G.path = require 'pl.path'
  _G.onmt = require 'onmt.init'
  _G.tds = require 'tds'
  -- if on-the-fly tokenization
  _G.separators = require('tools.utils.separators')
  _G.tokenizer = require('tools.utils.tokenizer')
  _G.BPE = require ('tools.utils.BPE')
  _G.bpes = {}
  _G.optTok = optTok
  _G.optMPr = optMPr
  _G.hookManager = onmt.utils.HookManager.new(args, id)
  _G.args = args
  for i, v in ipairs(optTok) do
    if v and v["bpe_model"] and v["bpe_model"] ~= '' then
      _G.bpes[i] = _G.BPE.new(v)
    end
  end
end

-- parse k=v options from distribution file, and build real key-value based on args
local function parseTextOptions(args, optList)
  local foptions = {}
  for _, o in ipairs(optList) do
    local kv = onmt.utils.String.split(o, "=")
    onmt.utils.Error.assert(#kv==1 or #kv==2, "incorrect option in distribution rules: "..o)
    -- boolean option
    if #kv == 1 then table.insert(kv, true) end
    onmt.utils.Error.assert(args[kv[1]] ~= nil, "option not defined in distribution rules: "..o)
    if type(args[kv[1]]) == "number" then kv[2]=tonumber(kv[2]) end
    if type(args[kv[1]]) == "boolean" then kv[2]=kv[2]~="" and kv[2]~="false" end
    foptions[kv[1]]=kv[2]
  end
  foptions.textOpt = table.concat(optList,";");
  return foptions
end

function Preprocessor:__init(args, dataType)
  tds = require('tds')

  self.dataType = dataType or 'bitext'
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, commonOptions)
  local options = declareDataOptions(self.dataType)
  for _, v in ipairs(commonOptions) do
    table.insert(options, v)
  end

  self.args = args

  local function isempty(t)
    local count = 0
    for _, v in ipairs(t) do
      if self.args[v] == '' then
        count = count + 1
      end
    end
    return count
  end

  -- tokenization and preprocessing options
  local optTok = { {}, {} }
  local optMPr = { {}, {} }
  for k, v in pairs(args) do
    if k:sub(1,4) == 'tok_' then
      local idx = 1
      if k:sub(5, 8) == 'tgt_' then
        idx = 2
        k = k:sub(9)
      elseif k:sub(5,8) == 'src_' then
        k = k:sub(9)
      else
        k = k:sub(5)
      end
      optTok[idx][k] = v
    end
    if k:sub(1,4) == 'mpr_' then
      local idx = 1
      if k:sub(5, 8) == 'tgt_' then
        idx = 2
        k = k:sub(9)
      elseif k:sub(5,8) == 'src_' then
        k = k:sub(9)
      else
        k = k:sub(5)
      end
      optMPr[idx][k] = v
    end
  end
  for i = 1, 2 do
    _G.logger:info("Using on-the-fly '%s' tokenization for input "..i, optTok[i]["mode"])
  end

  if args.preprocess_pthreads > 1 and args.train_dir ~= '' then
    local globalLogger = _G.logger
    -- try to load threads if available
    threads = require('threads')
    threads.Threads.serialization('threads.sharedserialize')
    self.pool = threads.Threads(
      args.preprocess_pthreads,
      function(id) init_thread(id, args, optTok, optMPr) end,
      function() _G.logger = globalLogger end
    )
  else
    init_thread("-", args, optTok, optMPr)
  end

  -- sanity check on options: train_dir is exclusive all direct file settings
  -- and for train_dir, we do need pre-build vocabulary
  if dataType == 'monotext' then
    self.trains = { 'train' }
    self.valids = { 'valid' }
    self.vocabs = { 'vocab' }
  else
    self.trains = { 'train_src', 'train_tgt' }
    self.valids = { 'valid_src', 'valid_tgt' }
    self.vocabs = { 'src_vocab', 'tgt_vocab' }
  end

  self.dist_rules = {}
  self.keep_rules = {}
  if args.gsample_dist ~= '' then
    local f = io.input(args.gsample_dist)
    while true do
      local dist_rule = f:read()
      if not dist_rule then break end
      local trule = onmt.utils.String.split(dist_rule, " ")
      local pattern = trule[1]
      local weight = trule[2]
      table.remove(trule,1)
      table.remove(trule,1)
      trule = { pattern, weight, parseTextOptions(args, trule) }
      if trule[2] == "*" then
        table.insert(self.keep_rules, trule)
      else
        table.insert(self.dist_rules, trule)
      end
    end
  end
  -- list and check training files
  if args.train_dir ~= '' then
    onmt.utils.Error.assert(isempty(self.trains) == #self.trains, 'For directory mode, file mode options (training) should not be set')
    if not args.dry_run then
      onmt.utils.Error.assert(isempty(self.vocabs) == 0, 'For directory mode, vocabs should be predefined')
    end
    self.totalCount, self.keepCount, self.list_train = self:parseDirectory(self.args, Preprocessor.getDataList(self.dataType), self.dist_rules, self.keep_rules, 'train')
  else
    onmt.utils.Error.assert(isempty(self.trains) == 0)
    self.totalCount = onmt.utils.FileReader.countLines(self.args[self.trains[1]], args.idx_files)
    self.keepCount = 0
    local list_files = { self.args[self.trains[1]] }
    for i = 2, #self.trains do
      table.insert(list_files, args[self.trains[i]])
      if not args.idx_files then
        onmt.utils.Error.assert(onmt.utils.FileReader.countLines(args[self.trains[i]], args.idx_files) == self.totalCount,
                                "line count in "..args[self.trains[i]].." do not match "..args[self.trains[1]])
      end
    end
    self.list_train = { { self.totalCount, list_files } }
    self.list_train[1].fname = self.args[self.trains[1]]
    self.list_train[1].weight = 1
    self.list_train[1].options = {}
  end

  if args[self.valids[1]] ~= '' then
    self.list_valid = { {onmt.utils.FileReader.countLines(args[self.valids[1]], args.idx_files), {args[self.valids[1]]}}}
    for i = 2, #self.valids do
      if not args.idx_files then
        onmt.utils.Error.assert(onmt.utils.FileReader.countLines(args[self.valids[i]], args.idx_files) == self.list_valid[1][1],
                                "line count in "..args[self.valids[i]].." do not match "..args[self.valids[1]])
      end
      table.insert(self.list_valid[1][2], args[self.valids[i]])
    end
    self.list_valid[1].fname = self.args[self.valids[1]]
    self.list_valid[1].weight = 1
    self.list_valid[1].options = {}
  end

end

--[[ Process on given tokenized sentence - check for validity and prepare structure ]]
local function processSentence(n, idx, tokens, parallelCheck, isValid, isInputVector, dicts,
                               constants, prunedRatio, generateFeatures, time_shift_feature,
                               sentenceDists, vectors, features, avgLength, sizes, allSizes,
                               src_seq_length, tgt_seq_length)
  local ignored = 0

  for i = 1, n do
    local length = (type(tokens[i])=='table' and #tokens[i]) or (tokens[i]:dim()==0 and 0) or tokens[i]:size(1)
    local idxRange = math.floor(length/10)+1
    if idxRange > #sentenceDists[i] then
      idxRange = #sentenceDists[i]
    end
    sentenceDists[i][idxRange] = sentenceDists[i][idxRange]+1
  end

  local valid = true

  if parallelCheck then
    valid = parallelCheck(idx, isInputVector, dicts, tokens)
  end

  for i = 1, n do
    -- if data is a Tensor, check for invalid values (NaN)
    if torch.typename(tokens[i]) and torch.typename(tokens[i]):find('torch.*Tensor') then
      if tokens[i]:ne(tokens[i]):sum() > 0 then
        valid=false
      end
    end
  end

  if valid and isValid(tokens, src_seq_length, tgt_seq_length) then
    for i = 1, n do

      local length = (type(tokens[i])=='table' and #tokens[i]) or (tokens[i]:dim()==0 and 0) or tokens[i]:size(1)
      avgLength[i] = avgLength[i] * (#allSizes[i] / (#allSizes[i] + 1)) + length / (#allSizes[i] + 1)

      if isInputVector[i] then
        vectors[i]:insert(tokens[i])
      else
        local words, feats = onmt.utils.Features.extract(tokens[i])
        local vocabs = onmt.utils.Placeholders.norm(words)
        local vec = dicts[i].words:convertToIdx(vocabs, table.unpack(constants[i]))
        local pruned = vec:eq(onmt.Constants.UNK):sum() / vec:size(1)

        prunedRatio[i] = prunedRatio[i] * (#allSizes[i] / (#allSizes[i] + 1)) + pruned / (#allSizes[i] + 1)
        vectors[i]:insert(vec)

        if not(isInputVector[i]) and #dicts[i].features > 0 then
          features[i]:insert(generateFeatures[i](dicts[i].features, feats, true, time_shift_feature))
        end
      end

      if i == 1 then
        sizes:insert(length)
      end
      allSizes[i]:insert(length)
    end
  else
    ignored = 1
  end

  return ignored
end

local function openDB_RW(path)
  if not lmdb then lmdb = require 'lmdb' end
  local db = lmdb.env {Path = path}
  db:open()
  local txn = db:txn()
  return db, txn
end

local function openDB_RO(path)
  -- setting bit as global; lmdb requires it; in some versions of lua, bit is not built-in.
  if not bit then bit = require 'bit' end
  if not lmdb then lmdb = require 'lmdb' end
  local db = lmdb.env {Path = path, RDONLY = true, MaxReaders = 126}
  db:open()
  local txn = db:txn(true)
  return db, txn
end

local function closeDB(db, txn)
  txn:commit()
  db:close()
end


--[[
  Generic data preparation function on multiples source
  * `files`: table of data source name
  * `isInputVector`: table of boolean indicating if corresponding source is an vector
  * `dicts`: table of dictionary data corresponding to source
  * `nameSources`: table of name of each source - for logging purpose
  * `constants`: constant to add to the vocabulary for each source
  * `isValid`: validation function taking prepared table of tokens from each source
  * `generateFeatures`: table of feature extraction fucnction for each source
  * `parallelCheck`: function to check parallely source/target(s)
  * `sample_file`: possible torch mapping vector
]]
function Preprocessor:makeGenericData(files, isInputVector, dicts, nameSources, constants,
                                      isValid, generateFeatures, parallelCheck, sample_file, dbNames)
  local verbose = _G.logger.level == 'DEBUG'
  sample_file = sample_file or {}
  local n = #files[1][2]

  if dbNames then
    onmt.utils.Error.assert(#dbNames == n,
       "Incorrect number of DB names (" .. tostring(dbNames) .. ") provided; should be ".. tostring(n))
  end

  local gSentenceDists = {}
  local gVectors = {}
  local gFeatures = {}
  local gAvgLength = {}
  local gSizes = tds.Vec()
  local gAllSizes = {}

  local gCount = 0
  local gIgnored = 0
  local gEmptyCount = 0

  for _ = 1, n do
    table.insert(gSentenceDists, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
    table.insert(gVectors, tds.Vec())
    table.insert(gFeatures, tds.Vec())
    table.insert(gAvgLength, 0)
    table.insert(gAllSizes, tds.Vec())
  end

  -- iterate on each file
  for _m, _df in ipairs(files) do
    self:poolAddJob(
      function(df, idx_files, audio_files, time_shift_feature, src_seq_length, tgt_seq_length, sampling)
        local count = 0
        local ignored = 0
        local emptyCount = 0

        local sentenceDists = {}
        local vectors = {}
        local features = {}
        local avgLength = {}
        local sizes = _G.tds.Vec()
        local allSizes = {}
        for i = 1, n do allSizes[i] = _G.tds.Vec() end

        local tmpDbNames = nil
        local db_main={}
        local txn_main={}
        local db_feat={}
        local txn_feat={}
        if dbNames and #dbNames > 0 then
          tmpDbNames = {}
          for i=1,n do
            table.insert(tmpDbNames, dbNames[i]..'-'.._G.__threadid)
            db_main[i], txn_main[i] = openDB_RW(tmpDbNames[i]..'-main-'..tostring(i))
            db_feat[i], txn_feat[i] = openDB_RW(tmpDbNames[i]..'-feat-'..tostring(i))
          end
        end
        local instanceId = 0

        for _ = 1, n do
          table.insert(sentenceDists, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
          table.insert(vectors, _G.tds.Vec())
          table.insert(features, _G.tds.Vec())
          table.insert(avgLength, 0)
        end

        -- if there is a sampling for this file
        local readers = {}
        local prunedRatio = {}
        for i = 1, n do
          table.insert(readers, onmt.utils.FileReader.new(df[2][i], idx_files, isInputVector[i] and not audio_files)) -- audio_files contain file name instead of vector
          table.insert(prunedRatio, 0)
        end

        --[[ if idx_files, indices in all files must have same order ]]
        if idx_files or audio_files then
          while true do
            local instance = {}
            local gIdx = nil; local endOfFile = false; local indices = {}
            --[[retrieve a instance of n-parallel tuple]]
            for i = 1, n do
              local tokens, idx = readers[i]:next()
              if not tokens then endOfFile = true; break end
              if idx_files then
                if gIdx then
                  if gIdx ~= idx then
                    return _G.__threadid, 1, string.format('%s Idx not defined in %s: '..idx, nameSources[i], nameSources[1])
                  end
                else
                  gIdx = idx
                end
              end
              if idx_files then
                if i == 1 and indices[gIdx] then
                  return _G.__threadid, 1, string.format('duplicate idx in %s file: '..gIdx, nameSources[i])
                end
                indices[gIdx] = true
              end

              if isInputVector[i] then
                if audio_files then
                  if _G.args.audio_feature_type == 'mfcc' then
                    instance[i] = onmt.data.Audio.getMfcc(tokens[1])
                  elseif _G.args.audio_feature_type == 'mfsc' then
                    instance[i] = onmt.data.Audio.getMfsc(tokens[1])
                  elseif _G.args.audio_feature_type == 'spectrogram' then
                    instance[i] = onmt.data.Audio.getSpectrogram(tokens[1])
                  else
                    _G.logger:error('Unknown audio feature type: %s', _G.args.audio_feature_type)
                    os.exit(1)
                  end
                else
                  instance[i] = torch.Tensor(tokens)
                end
              else
                instance[i] = tokens
              end
            end

            if endOfFile then break end

            ignored = ignored + processSentence(n, instanceId, instance, parallelCheck, isValid, isInputVector, dicts,
                                                constants, prunedRatio, generateFeatures, time_shift_feature,
                                                sentenceDists, vectors, features, avgLength, sizes, allSizes,
                                                src_seq_length, tgt_seq_length)

              --[[ move one instance tuple that was just created from above function processSentenece into lmdb ]]
              if tmpDbNames and #tmpDbNames > 0 and #vectors[1] > 0 then
                instanceId = instanceId + 1
                for i=1, n do
                  txn_main[i]:put(instanceId, vectors[i][1])
                  vectors[i]:remove(1)
                  if features[i] and #features[i] > 0 then
                    txn_feat[i]:put(instanceId, features[i][1])
                    features[i]:remove(1)
                  end
                end
                if instanceId % 500 == 0 then
                  for i=1, n do
                    txn_main[i]:commit(); txn_main[i] = db_main[i]:txn()
                    txn_feat[i]:commit(); txn_feat[i] = db_feat[i]:txn()
                  end
                  collectgarbage()
                end
              end

            count = count + 1
          end
        else
          local idx = 1
          local sampling_idx = 1
          local hasNil = false
          -- read all the available sentences or as long as we have not reached sampling size
          -- sampling table is an ordered sentence of sentences id to keep
          while not hasNil and (not sampling or sampling_idx <= #sampling) do
            -- keep in sentences the different sentences and number of times it repeats
            local sentences = { {} }
            for _ = 1, n do
              table.insert(sentences, {})
            end
            -- keep maximum a batch of 10000 sentences
            while not hasNil and (not sampling or sampling_idx <= #sampling) and #sentences[1] < preprocess_batchsize do
              local allNil = true
              local keepSentence = not sampling or sampling[sampling_idx] == idx

              for i = 1, n do
                local sentence = readers[i]:next(false)
                hasNil = hasNil or sentence == nil
                allNil = allNil and sentence == nil
                if sentence and keepSentence then
                  table.insert(sentences[i+1], sentence)
                end
              end

              if hasNil then
                if not allNil then
                  return _G.__threadid, 1, string.format('all data sources do not have the same number of sentences')
                end
                break
              end

              local repeatSentence = 1
              if sampling then
                while sampling_idx+repeatSentence <= #sampling and sampling[sampling_idx+repeatSentence] == idx do
                  repeatSentence = repeatSentence + 1
                end
              end

              if keepSentence then
                if sampling then
                  sampling_idx = sampling_idx + repeatSentence
                end
                table.insert(sentences[1], repeatSentence)
              end
              idx = idx + 1
            end

            if #sentences[1] > 0 then
              -- preprocess and tokenize
              for i = 1, n do
                -- adapt local options
                local savOpt = {}
                for k, v in pairs(df.options or {}) do
                  savOpt[k] = _G.optMPr[i][k]
                  _G.optMPr[i][k] = v
                end
                local psentences = _G.hookManager:call("mpreprocess", _G.optMPr[i], sentences[i+1])
                if psentences then
                  sentences[i+1] = psentences
                end
                -- restore options
                for k, _ in pairs(df.options or {}) do
                  _G.optMPr[i][k] = savOpt[k]
                end
              end

              if n == 2 then
                local asentences = { sentences[2], sentences[3] }
                -- adapt local options
                local savOpt = {}
                for k, v in pairs(df.options or {}) do
                  savOpt[k] = _G.args[k]
                  _G.args[k] = v
                end
                local psentences = _G.hookManager:call("bpreprocess", _G.args, asentences)
                if psentences then
                  if verbose then
                    _G.logger:info("bpreprocess results: %d remaining out of %d", #psentences[1], #sentences[2])
                  end
                  sentences[2] = psentences[1]
                  sentences[3] = psentences[2]
                end
                -- restore options
                for k, _ in pairs(df.options or {}) do
                  _G.args[k] = savOpt[k]
                end
              end

              for i = 1, n do
                for j = 1, #sentences[i+1] do
                  sentences[i+1][j] =  _G.tokenizer.tokenize(_G.optTok[i], sentences[i+1][j], _G.bpes[i])
                end
              end

              for j = 1, #sentences[2] do
                local tokens = {}
                for i = 1, n do
                  table.insert(tokens, sentences[i+1][j])
                  if verbose then
                    _G.logger:debug("[%d:%d] %s", j, i, table.concat(tokens[i], " "))
                  end
                end
                for _ = 1, sentences[1][j] do
                  ignored = ignored + processSentence(n, idx, tokens, parallelCheck, isValid, isInputVector, dicts,
                                                          constants, prunedRatio, generateFeatures, time_shift_feature,
                                                          sentenceDists, vectors, features, avgLength, sizes, allSizes,
                                                          src_seq_length, tgt_seq_length)
                  count = count + 1

                  --[[ move one instance tuple that was just created from above function processSentenece into lmdb ]]
                  if tmpDbNames and #tmpDbNames > 0 and #vectors[1] > 0 then
                    instanceId = instanceId + 1
                    for i=1, n do
                      txn_main[i]:put(instanceId, vectors[i][1])
                      vectors[i]:remove(1)
                      if #features[i] > 0 then
                        txn_feat[i]:put(instanceId, features[i][1])
                        features[i]:remove(1)
                      end
                    end
                    if instanceId % 500 == 0 then
                      for i=1, n do
                        txn_main[i]:commit(); txn_main[i] = db_main[i]:txn()
                        txn_feat[i]:commit(); txn_feat[i] = db_feat[i]:txn()
                      end
                      collectgarbage()
                    end
                  end
                end
              end
            end
          end
        end

        for i = 1, n do
          readers[i]:close()
        end

        if tmpDbNames and #tmpDbNames > 0 then
          for i=1,n do
            closeDB(db_main[i], txn_main[i])
            closeDB(db_feat[i], txn_feat[i])
          end
        end

        return _G.__threadid, false, sentenceDists, vectors, features, avgLength, sizes, allSizes, prunedRatio, count, ignored, emptyCount,
               sampling and #sampling or _df[1], tmpDbNames
      end,
      -- aggregate the results together
      function(__threadid, error, sentenceDists, vectors, features, avgLength, sizes, allSizes, prunedRatio, count, ignored, emptyCount, kept, tmpDbNames)
        if error then
          _G.logger:error(sentenceDists)
          os.exit(1)
        end

        local instanceID
        local g_db_main = {}
        local g_txn_main = {}
        local g_db_feat = {}
        local g_txn_feat = {}
        if dbNames and #dbNames > 0 then
          for i=1, n do
            g_db_main[i], g_txn_main[i] = openDB_RW(dbNames[i] .. '-main.lmdb')
            g_db_feat[i], g_txn_feat[i] = openDB_RW(dbNames[i] .. '-feat.lmdb')
          end
        end

        for i = 1, n do
          for j=1, #gSentenceDists[i] do
            gSentenceDists[i][j] = gSentenceDists[i][j] + sentenceDists[i][j]
          end

          if dbNames and #dbNames > 0 then

            instanceID = g_db_main[i]:stat()['entries']

            local tmp_db_main, tmp_txn_main = openDB_RO(tmpDbNames[i]..'-main-'..tostring(i))
            local tmp_db_feat, tmp_txn_feat = openDB_RO(tmpDbNames[i]..'-feat-'..tostring(i))
            local tmpMainDbCnt = tmp_db_main:stat()['entries']
            local tmpFeatDbCnt = tmp_db_feat:stat()['entries']
            if tmpFeatDbCnt > 0 then
              assert(tmpMainDbCnt == tmpFeatDbCnt)
            end
            for tmpDbIdx = 1, tmpMainDbCnt do
              instanceID = instanceID + 1
              g_txn_main[i]:put(instanceID, tmp_txn_main:get(tmpDbIdx))
              if tmpFeatDbCnt > 0 then
                g_txn_feat[i]:put(instanceID, tmp_txn_feat:get(tmpDbIdx))
              end
              if instanceID % 500 == 0 then
                g_txn_main[i]:commit(); g_txn_main[i] = g_db_main[i]:txn()
                g_txn_feat[i]:commit(); g_txn_feat[i] = g_db_feat[i]:txn()
                collectgarbage()
              end
            end
            closeDB(tmp_db_main, tmp_txn_main)
            closeDB(tmp_db_feat, tmp_txn_feat)
          else
            for j=1, #vectors[i] do
              gVectors[i]:insert(vectors[i][j])
            end
            for j=1, #features[i] do
              gFeatures[i]:insert(features[i][j])
            end
          end

          for j = 1, #allSizes[i] do
            gAllSizes[i]:insert(allSizes[i][j])
          end

          gAvgLength[i] = (gAvgLength[i] * (#gAllSizes[i]-#allSizes[i]) + avgLength[i] * #allSizes[i])/#gAllSizes[i]

        end
        for j=1, #sizes do
          gSizes:insert(sizes[j])
        end
        local msgPrune = ''
        for i = 1, n do
          msgPrune = msgPrune .. (i==1 and '' or ', ')
          msgPrune = msgPrune .. nameSources[i] .. ' = '..string.format("%.1f%%", prunedRatio[i] * 100)
        end

        if dbNames and #dbNames > 0 then
          for i=1,n do
            closeDB(g_db_main[i], g_txn_main[i])
            closeDB(g_db_feat[i], g_txn_feat[i])
            os.execute('rm -rd "'..tmpDbNames[i]..'-main-'..tostring(i)..'"')
            os.execute('rm -rd "'..tmpDbNames[i]..'-feat-'..tostring(i)..'"')
          end
        end

        _G.logger:info(' * ['..__threadid..'] file \'%s\' (%s): %d total, %d drawn, %d kept - unknown words: %s',
                          _df.fname or "n/a", (_df.options and _df.options.textOpt) or '', _df[1], kept, #allSizes[1], msgPrune)

        gCount = gCount + count
        gIgnored = gIgnored + ignored
        gEmptyCount = gEmptyCount + emptyCount
      end,
      _df, self.args.idx_files, self.dataType == 'audiotext', self.args.time_shift_feature, self.args.src_seq_length or self.args.seq_length, self.args.tgt_seq_length, sample_file[_m])
  end

  self:poolSynchronize()

  for i = 1, n do
    for j=1, #gSentenceDists[i] do
      gSentenceDists[i][j] = gSentenceDists[i][j] / gCount
    end
  end

  local function reorderData(perm)
    for i = 1, n do
      gVectors[i] = onmt.utils.Table.reorder(gVectors[i], perm, true, isInputVector and isInputVector[i])
      if not(isInputVector[i]) and #dicts[i].features > 0 then
        gFeatures[i] = onmt.utils.Table.reorder(gFeatures[i], perm, true)
      end
    end
  end

  onmt.utils.Error.assert(#gAllSizes[1] > 0, "empty dataset")

  if self.args.shuffle and not self.args.lmdb then
    _G.logger:info('... shuffling sentences')
    local perm = torch.randperm(#gVectors[1])
    gSizes = onmt.utils.Table.reorder(gSizes, perm, true)
    reorderData(perm)
  end

  if self.args.sort and not self.args.lmdb then
    _G.logger:info('... sorting sentences by size')
    local _, perm = torch.sort(vecToTensor(gSizes))
    reorderData(perm)
  end

  _G.logger:info('Prepared %d sentences:', #gSizes)
  _G.logger:info(' * %d sequences not validated (length, other)', gIgnored)
  local msgLength = ''
  for i = 1, n do
    msgLength = msgLength .. (i==1 and '' or ', ')
    msgLength = msgLength .. nameSources[i] .. ' = '..string.format("%.1f", gAvgLength[i])
  end

  _G.logger:info(' * average sequence length: '..msgLength)

  local data = {}

  for i = 1, n do
    local dist='[ '
    for j = 1, #gSentenceDists[1] do
      if j>1 then
        dist = dist..' ; '
      end
      dist = dist..math.floor(gSentenceDists[i][j]*100)..'%%'
    end
    dist = dist..' ]'
    _G.logger:info(' * %s sentence length (range of 10): '..dist, nameSources[i])


    local mainEntry
    local featEntry
    if dbNames and #dbNames > 0 then
      mainEntry = dbNames[i] .. '-main.lmdb'
      featEntry = dbNames[i] .. '-feat.lmdb'
    else
      mainEntry = gVectors[i]
      featEntry = gFeatures[i]
    end

    if isInputVector[i] then
      table.insert(data, { vectors = mainEntry, features = featEntry })
    else
      table.insert(data, { words = mainEntry, features = featEntry })
    end

  end

  return data
end

--[[ Check data validity ]]
local function isValid(seq, maxSeqLength)
  if torch.isTensor(seq) then
    return seq:size(1) > 0 and seq:size(1) <= maxSeqLength
  end
  return #seq > 0 and #seq <= maxSeqLength
end

local function validBilingual(tokens, src_seq_length, tgt_seq_length)
  return #tokens[1] > 0 and
         isValid(tokens[1], src_seq_length) and
         #tokens[2] > 0 and
         isValid(tokens[2], tgt_seq_length)
end

function Preprocessor:makeBilingualData(files, srcDicts, tgtDicts, sample_file, dbPrefix)
  local data = self:makeGenericData(
                              files,
                              { false, false },
                              { srcDicts, tgtDicts },
                              { 'source', 'target' },
                              {
                                {
                                  onmt.Constants.UNK_WORD
                                },
                                {
                                  onmt.Constants.UNK_WORD,
                                  onmt.Constants.BOS_WORD,
                                  onmt.Constants.EOS_WORD
                                }
                              },
                              validBilingual,
                              {
                                onmt.utils.Features.generateSource,
                                onmt.utils.Features.generateTarget
                              },
                              self.args.check_plength and self.parallelCheck,
                              sample_file,
                              {dbPrefix .. '-src', dbPrefix .. '-tgt'})
  return data[1], data[2]
end

local function validFeat(tokens, src_seq_length, tgt_seq_length)
  return tokens[1]:dim() > 0 and
         isValid(tokens[1], src_seq_length) and
         #tokens[2] > 0 and
         isValid(tokens[2], tgt_seq_length)
end

function Preprocessor:makeFeatTextData(files, tgtDicts, sample_file, dbPrefix)
  local data = self:makeGenericData(
                              files,
                              { true, false },
                              { {}, tgtDicts },
                              { 'source', 'target' },
                              {
                                false,
                                {
                                  onmt.Constants.UNK_WORD,
                                  onmt.Constants.BOS_WORD,
                                  onmt.Constants.EOS_WORD
                                }
                              },
                              validFeat,
                              {
                                false,
                                onmt.utils.Features.generateTarget
                              },
                              self.args.check_plength and self.parallelCheck,
                              sample_file,
                              {dbPrefix .. '-src', dbPrefix .. '-tgt'})
  return data[1], data[2]
end

function Preprocessor:makeAudioTextData(files, tgtDicts, sample_file, dbPrefix)
  local data = self:makeGenericData(
                              files,                       -- * `files`: table of data source name
                              { true, false },             -- * `isInputVector`: table of boolean indicating if corresponding source is an vector
                              { {}, tgtDicts },            -- * `dicts`: table of dictionary data corresponding to source
                              { 'source', 'target' },      -- * `nameSources`: table of name of each source - for logging purpose
                              {                            -- * `constants`: constant to add to the vocabulary for each source
                                false,
                                {
                                  onmt.Constants.UNK_WORD,
                                  onmt.Constants.BOS_WORD,
                                  onmt.Constants.EOS_WORD
                                }
                              },
                              validFeat,                   -- * `isValid`: validation function taking prepared table of tokens from each source
                              {                            -- * `generateFeatures`: table of feature extraction fucnction for each source
                                false,
                                onmt.utils.Features.generateTarget
                              },
                              self.args.check_plength and self.parallelCheck, -- * `parallelCheck`: function to check parallely source/target(s)
                              sample_file,                -- * `sample_file`: possible torch mapping vector
                              {dbPrefix .. '-src', dbPrefix .. '-tgt'})
  return data[1], data[2]
end

local function ValidMono(tokens, seq_length)
  return #tokens[1] > 0 and isValid(tokens[1], seq_length)
end

function Preprocessor:makeMonolingualData(files, dicts, sample_file, dbPrefix)
  local data = self:makeGenericData(
                              files,
                              { false },
                              { dicts },
                              { 'source' },
                              {
                                {
                                  onmt.Constants.UNK_WORD,
                                  onmt.Constants.BOS_WORD,
                                  onmt.Constants.EOS_WORD
                                }
                              },
                              ValidMono,
                              {
                                onmt.utils.Features.generateTarget
                              },
                              nil,
                              sample_file,
                              {dbPrefix .. '-src'})
  return data[1]
end

function Preprocessor.parallelCheck(idx, _, _, tokens)
  local length1 = (type(tokens[1])=='table' and #tokens[1]) or (tokens[1]:dim()==0 and 0) or tokens[1]:size(1)
  local length2 = (type(tokens[2])=='table' and #tokens[2]) or (tokens[2]:dim()==0 and 0) or tokens[2]:size(1)
  if length1~=length2 then
    _G.logger:warning('SENT %s: source/target not aligned (%d/%d)', tostring(idx), length1, length2)
    return false
  end
  return true
end

function Preprocessor:getVocabulary()
  local dicts = {}
  -- use the first source file to count source features
  local src_file = self.list_train[1][2][1]
  if self.dataType ~= 'feattext' and self.dataType ~= 'audiotext' then
    dicts.src = onmt.data.Vocabulary.init('source',
                                     src_file,
                                     self.args.src_vocab or self.args.vocab,
                                     self.args.src_vocab_size or self.args.vocab_size,
                                     self.args.src_words_min_frequency or self.args.words_min_frequency,
                                     self.args.features_vocabs_prefix,
                                     function(s) return isValid(s, self.args.src_seq_length or self.args.seq_length) end,
                                     self.args.keep_frequency,
                                     self.args.idx_files,
                                     self.args.tok_src_case_feature)
  end
  if self.dataType ~= 'monotext' then
    -- use the first target file to count target features
    local tgt_file = self.list_train[1][2][2]
    dicts.tgt = onmt.data.Vocabulary.init('target',
                                     tgt_file,
                                     self.args.tgt_vocab,
                                     self.args.tgt_vocab_size,
                                     self.args.tgt_words_min_frequency,
                                     self.args.features_vocabs_prefix,
                                     function(s) return isValid(s, self.args.tgt_seq_length) end,
                                     self.args.keep_frequency,
                                     self.args.idx_files,
                                     self.args.tok_tgt_case_feature)
  end
  return dicts
end

function Preprocessor:makeData(dataset, dicts, dbPrefix)

  if dataset ~= 'valid' or
     (self.args.valid and self.args.valid ~= '') or
     (self.args.valid_src and self.args.valid_src ~= '') or
     (self.args.valid_tgt and self.args.valid_tgt ~= '') then

    _G.logger:info("--- Preparing "..dataset.." sample")

    local sample_file = {}
    if dataset == 'train' and self.args.gsample ~= 0 then
      -- sample data using sample and sample_dict
      local sampledCount = self.args.gsample
      if sampledCount < 1 then
        sampledCount = sampledCount * self.totalCount
      else
        sampledCount = sampledCount - self.keepCount
      end
      if self.totalCount == 0 and self.keepCount < self.args.gsample then
        _G.logger:warning('You requested a sample of %d sentences but no files matched any sampling rules and only %d sentences are selected by keep rules. There could be issues with your distribution rules.',
                          self.args.gsample, self.keepCount)
      end
      if sampledCount < 0 then
        _G.logger:error('You requested a sample of %d sentences but %d are already reserved by keep rules. You should configure a larger sample or keep less sentences.',
                        self.args.gsample, self.keepCount)
        os.exit(1)
      end
      -- check how many sentences per file
      for _, f in ipairs(self.list_train) do
        if f.weight == math.huge then
          table.insert(sample_file, tds.Vec(torch.range(1, f[1]):totable()))
        else
          local n = math.ceil(sampledCount * f.weight)
          local t = torch.LongTensor(n)
          if n > 0 then
            for i = 1, n do
              t[i] = torch.random(1, f[1])
            end
            t = torch.sort(t)
          end
          table.insert(sample_file, tds.Vec(t:totable()))
        end
      end
    end

    if not self.args.lmdb then
      dbPrefix = nil
    end

    local data = {}
    if self.dataType == 'monotext' then
      data.src = self:makeMonolingualData(self["list_"..dataset], dicts.src, sample_file, dbPrefix)
    elseif self.dataType == 'feattext' then
      data.src, data.tgt = self:makeFeatTextData(self["list_"..dataset], dicts.tgt, sample_file, dbPrefix)
      if not dicts.srcInputSize then
        dicts.srcInputSize = data.src.vectors[1]:size(2)
      else
        onmt.utils.Error.assert(dicts.srcInputSize==data.src.vectors[1]:size(2), "feature size is not matching in all files")
      end
    elseif self.dataType == 'audiotext' then
      data.src, data.tgt = self:makeAudioTextData(self["list_"..dataset], dicts.tgt, sample_file, dbPrefix)

      local srcInputSize
      if type(data.src.vectors) == 'string' then
        local db, txn = openDB_RO(data.src.vectors)
        srcInputSize = txn:get(1):size(2)
        closeDB(db, txn)
      else
        srcInputSize = data.src.vectors[1]:size(2)
      end

      if not dicts.srcInputSize then
        dicts.srcInputSize = srcInputSize
      else
        onmt.utils.Error.assert(dicts.srcInputSize==srcInputSize, "feature size is not matching in all files")
      end
    else
      data.src, data.tgt = self:makeBilingualData(self["list_"..dataset], dicts.src, dicts.tgt, sample_file, dbPrefix)
    end

    _G.logger:info("")

    return data
  else
    _G.logger:warning('No validation data')
  end
end

return Preprocessor
