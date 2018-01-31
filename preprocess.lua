require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('preprocess.lua')

-- First argument define the dataType: bitext/monotext - default is bitext.
local dataType = cmd.getArgument(arg, '-data_type') or 'bitext'

-- Options declaration
local options = {
  {
    '-data_type', 'bitext',
    [[Type of data to preprocess. Use 'monotext' for monolingual data.
      This option impacts all options choices.]],
    {
      enum = {'bitext', 'monotext', 'feattext', 'audiotext'},
      depends = function(opt) return opt.data_type ~= 'feattext' or opt.idx_files end
    }
  },
  {
    '-dry_run', false,
    [[If set, this will only prepare the preprocessor. Useful when using file sampling to
      test distribution rules.]]
  },
  {
    '-save_data', '',
    [[Output file for the prepared data.]],
    {
      depends = function(opt)
        return opt.dry_run or opt.save_data ~= '', "option `-save_data` is required"
      end
    }
  },
  {
    '-lmdb', false,
    [[If set, save src and tgt tables in train and valid as lmdb.]]
  }
}

cmd:setCmdLineOptions(options, 'Preprocess')

onmt.data.Preprocessor.declareOpts(cmd, dataType)
-- insert on the fly the option depending if there is a hook selected
onmt.utils.HookManager.updateOpt(arg, cmd)

-- expand options depending on source or target (tokenization, mpreprocessing)
onmt.data.Preprocessor.expandOpts(cmd, dataType)

onmt.utils.HookManager.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local otherOptions = {
  {
    '-seed', 3425,
    [[Random seed.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  }
}
cmd:setCmdLineOptions(otherOptions, 'Other')

local opt = cmd:parse(arg)

local function main()

  torch.manualSeed(opt.seed)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  _G.hookManager = onmt.utils.HookManager.new(opt)

  local Preprocessor = onmt.data.Preprocessor.new(opt, dataType)

  if opt.dry_run then
    _G.logger:shutDown()
    return
  end

  local data = { dataType=dataType }

  -- keep processing options in the structure for further traceability
  data.opt = opt

  _G.logger:info('Preparing vocabulary...')
  data.dicts = Preprocessor:getVocabulary()

  _G.logger:info('Preparing training data...')
  data.train = Preprocessor:makeData('train', data.dicts)
  _G.logger:info('')

  _G.logger:info('Preparing validation data...')
  data.valid = Preprocessor:makeData('valid', data.dicts)
  _G.logger:info('')

  if dataType == 'monotext' then
    if opt.vocab:len() == 0 then
      onmt.data.Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      onmt.data.Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data)
    end
  elseif dataType == 'feattext' or dataType == 'audiotext' then
    if opt.tgt_vocab:len() == 0 then
      onmt.data.Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      onmt.data.Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data)
    end
  else
    if opt.src_vocab:len() == 0 then
      onmt.data.Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.src.dict')
    end

    if opt.tgt_vocab:len() == 0 then
      onmt.data.Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      onmt.data.Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data..'.source')
      onmt.data.Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data..'.target')
    end
  end

  _G.logger:info('Saving data to \'' .. opt.save_data .. '-train.t7\'...')

  if opt.lmdb then
    local lmdb = require('lmdb')
    local function toLMDB(dbName, tbl)
      local function openDB_RW(path, name)
        local db = lmdb.env {Path = path, Name = name}
        db:open()
        local txn = db:txn()
        return db, txn
      end
      local function closeDB(db, txn)
        txn:commit()
        db:close()
      end
      local db, txn = openDB_RW(dbName, dbName)
      for idx, entry in ipairs(tbl) do
        txn:put(idx, entry)
        if idx % 500 == 0 then
          txn:commit(); txn = db:txn()
          collectgarbage()
        end
      end
      txn:commit(); txn = db:txn()
        _G.logger:info('From ' .. #tbl .. ', saved ' .. tostring(db:stat()['entries']) .. ' data to \'' .. dbName)
      closeDB(db, txn)
      return dbName
    end

    local function saveDataAsLmdb(data, prefix)
      if not data then
        return data
      end
      if data.words then
        data.words = toLMDB(prefix .. '-main.lmdb', data.words)
      elseif data.vectors then
        data.vectors = toLMDB(prefix .. '-main.lmdb', data.vectors)
      else
        _G.logger:error('Missing main src feature in train')
      end
      if data.features then
        data.features = toLMDB(prefix .. '-feat.lmdb', data.features)
      end
    end

    if data.train then
      saveDataAsLmdb(data.train.src, opt.save_data .. '-train-src')
      saveDataAsLmdb(data.train.tgt, opt.save_data .. '-train-tgt')
    end
    if data.valid then
      saveDataAsLmdb(data.valid.src, opt.save_data .. '-valid-src')
      saveDataAsLmdb(data.valid.tgt, opt.save_data .. '-valid-tgt')
    end
  end

  torch.save(opt.save_data .. '-train.t7', data, 'binary', false)
  _G.logger:shutDown()
end

main()
