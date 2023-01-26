import pandas as pd


def languages_to_features(languages_list):
    from sklearn.decomposition import PCA
    import lang2vec.lang2vec as l2v
    features_list = ["syntax_knn", "phonology_knn", "inventory_knn"]
    features = l2v.get_features(languages_list, features_list)

    features_matrix = []
    for language in languages_list:
        features_matrix.append(features[language])

    pca = PCA(n_components=6)
    pca_features_matrix = pca.fit_transform(features_matrix)
    res = {}

    for i, language in enumerate(languages_list):
        res[languages_list[i]] = pca_features_matrix[i]
    return res


def get_language_pair_side_info(data_dir, side_info_features, src_lang_name, tgt_lang_name):
    data = pd.read_csv(data_dir)
    side_dict = {}
    for record in data.iterrows():
        record = record[1]
        src_lang = record[src_lang_name]
        tgt_lang = record[tgt_lang_name]
        if side_info_features is not None:
            side_dict[src_lang + "_" + tgt_lang] = record[side_info_features].values
        else:
            side_dict[src_lang + "_" + tgt_lang] = None
    return side_dict


def get_language_side_information(data_dir, src_lang_name, tgt_lang_name):
    data = pd.read_csv(data_dir)
    side_dict = {}
    src_langs = set()
    tgt_langs = set()

    for record in data.iterrows():
        record = record[1]
        src_lang = record[src_lang_name]
        tgt_lang = record[tgt_lang_name]
        src_langs.add(src_lang)
        tgt_langs.add(tgt_lang)

    src_langs = list(src_langs)
    tgt_langs = list(tgt_langs)

    return languages_to_features(src_langs), languages_to_features(tgt_langs)


def get_info(task, attributes, ctx, params, DIR):

    if task == 'wiki_MF':
        SRC = "Source"
        TGT = "Target"
        SCORE = ["BLEU"]
        reg_x = [None]
        reg_y = [None]
        reg_z = [None]
        tgt_si_len = 0
        src_si_len = 0
        lang_pair_si_len = 0
        SIDE_FEATURES = None
        CONTEXT_FEATURES = None
        SIDE_INFO_DICT = None
        SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None
        ctx = None
        attributes = None

    elif task == 'bli_MF':
        SRC = "Source Language Code"
        TGT = "Target Language Code"
        SCORE = ["Vecmap", "Muse"]
        reg_x = [None]
        reg_y = [None]
        reg_z = [None]
        tgt_si_len = 0
        src_si_len = 0
        lang_pair_si_len = 0
        SIDE_FEATURES = None
        CONTEXT_FEATURES = None
        SIDE_INFO_DICT = None
        SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None
        ctx = None
        attributes = None

    elif task == 'tsfpos_MF':
        SRC = "Task lang"
        TGT = "Aux lang"
        SCORE = ["Accuracy"]
        reg_x = [None]
        reg_y = [None]
        reg_z = [None]
        tgt_si_len = 0
        src_si_len = 0
        lang_pair_si_len = 0
        SIDE_FEATURES = None
        CONTEXT_FEATURES = None
        SIDE_INFO_DICT = None
        SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None
        ctx = None
        attributes = None

    elif task == 'tsfel_MF':
        SRC = "Target lang"
        TGT = "Transfer lang"
        SCORE = ["Accuracy"]
        reg_x = [None]
        reg_y = [None]
        reg_z = [None]
        tgt_si_len = 0
        src_si_len = 0
        lang_pair_si_len = 0
        SIDE_FEATURES = None
        CONTEXT_FEATURES = None
        SIDE_INFO_DICT = None
        SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None
        ctx = None
        attributes = None

    elif task == 'tsfmt_MF':
        SRC = " Source lang"
        TGT = "Transfer lang"
        SCORE = ["BLEU"]
        reg_x = [None]
        reg_y = [None]
        reg_z = [None]
        tgt_si_len = 0
        src_si_len = 0
        lang_pair_si_len = 0
        SIDE_FEATURES = None
        CONTEXT_FEATURES = None
        SIDE_INFO_DICT = None
        SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None
        ctx = None
        attributes = None

    elif task == 'tsfparsing_MF':
        SRC = "Target lang"
        TGT = "Transfer lang"
        SCORE = ["Accuracy"]
        reg_x = [None]
        reg_y = [None]
        reg_z = [None]
        tgt_si_len = 0
        src_si_len = 0
        lang_pair_si_len = 0
        SIDE_FEATURES = None
        CONTEXT_FEATURES = None
        SIDE_INFO_DICT = None
        SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None
        ctx = None
        attributes = None

    elif task == 'tsfpos_aaMF':
        SRC = "Task lang"
        TGT = "Aux lang"
        SCORE = ["Accuracy"]

        if attributes == 'ctx':
            SIDE_FEATURES = None
            CONTEXT_FEATURES = get_context(ctx, task)
            lang_pair_si_len = len(CONTEXT_FEATURES)
            tgt_si_len = 0
            src_si_len = 0
            reg_x = [None]
            reg_y = [None]
            reg_z = params.reg_z
            SIDE_INFO_DICT = get_language_pair_side_info(DIR, CONTEXT_FEATURES, SRC, TGT)
            SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None


    elif task == 'tsfel_aaMF':
        SRC = "Target lang"
        TGT = "Transfer lang"
        SCORE = ["Accuracy"]

        if attributes == 'ctx':
            SIDE_FEATURES = None
            CONTEXT_FEATURES = get_context(ctx, task)
            lang_pair_si_len = len(CONTEXT_FEATURES)
            tgt_si_len = 0
            src_si_len = 0
            reg_x = [None]
            reg_y = [None]
            reg_z = params.reg_z
            SIDE_INFO_DICT = get_language_pair_side_info(DIR, CONTEXT_FEATURES, SRC, TGT)
            SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None


    elif task == 'tsfmt_aaMF':
        SRC = " Source lang"
        TGT = "Transfer lang"
        SCORE = ["BLEU"]

        if attributes == 'ctx':
            SIDE_FEATURES = None
            CONTEXT_FEATURES = get_context(ctx, task)
            lang_pair_si_len = len(CONTEXT_FEATURES)
            tgt_si_len = 0
            src_si_len = 0
            reg_x = [None]
            reg_y = [None]
            reg_z = params.reg_z
            SIDE_INFO_DICT = get_language_pair_side_info(DIR, CONTEXT_FEATURES, SRC, TGT)
            SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None


    elif task == 'tsfparsing_aaMF':
        SRC = "Target lang"
        TGT = "Transfer lang"
        SCORE = ["Accuracy"]

        if attributes == 'ctx':
            SIDE_FEATURES = None
            CONTEXT_FEATURES = get_context(ctx, task)
            lang_pair_si_len = len(CONTEXT_FEATURES)
            tgt_si_len = 0
            src_si_len = 0
            reg_x = [None]
            reg_y = [None]
            reg_z = params.reg_z
            SIDE_INFO_DICT = get_language_pair_side_info(DIR, CONTEXT_FEATURES, SRC, TGT)
            SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None


    elif task == 'wiki_aaMF':
        SRC = "Source"
        TGT = "Target"
        SCORE = ["BLEU"]


        if attributes == 'si':
            SIDE_FEATURES = ['geographic', 'genetic', 'inventory', 'syntactic', 'phonological', 'featural']
            CONTEXT_FEATURES = None
            tgt_si_len = len(SIDE_FEATURES)
            src_si_len = len(SIDE_FEATURES)
            lang_pair_si_len = 0
            reg_x = params.reg_x
            reg_y = params.reg_y
            reg_z = [None]
            SIDE_INFO_DICT = None
            SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = get_language_side_information(DIR, SRC, TGT)
            ctx = None


        elif attributes == 'ctx':
            SIDE_FEATURES = None
            CONTEXT_FEATURES = get_context(ctx, task)
            lang_pair_si_len = len(CONTEXT_FEATURES)
            tgt_si_len = 0
            src_si_len = 0
            reg_x = [None]
            reg_y = [None]
            reg_z = params.reg_z
            SIDE_INFO_DICT = get_language_pair_side_info(DIR, CONTEXT_FEATURES, SRC, TGT)
            SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None

        elif attributes == 'csi':
            SIDE_FEATURES = ['geographic', 'genetic', 'inventory', 'syntactic', 'phonological', 'featural']
            CONTEXT_FEATURES = get_context(ctx, task)
            tgt_si_len = len(SIDE_FEATURES)
            src_si_len = len(SIDE_FEATURES)
            lang_pair_si_len = len(CONTEXT_FEATURES)
            reg_x = params.reg_x
            reg_y = params.reg_y
            reg_z = params.reg_z
            SIDE_INFO_DICT = get_language_pair_side_info(DIR, CONTEXT_FEATURES, SRC, TGT)
            SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = get_language_side_information(DIR, SRC, TGT)

    elif task == 'bli_aaMF':
        SRC = "Source Language Code"
        TGT = "Target Language Code"
        SCORE = ["Vecmap", "Muse"]


        if attributes == 'si':
            SIDE_FEATURES = ['geographic', 'genetic', 'inventory', 'syntactic', 'phonological', 'featural']
            CONTEXT_FEATURES = None
            tgt_si_len = len(SIDE_FEATURES)
            src_si_len = len(SIDE_FEATURES)
            lang_pair_si_len = 0
            reg_x = params.reg_x
            reg_y = params.reg_y
            reg_z = [None]
            SIDE_INFO_DICT = None
            SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = get_language_side_information(DIR, SRC, TGT)
            ctx = None

        elif attributes == 'csi':
            SIDE_FEATURES = ['geographic', 'genetic', 'inventory', 'syntactic', 'phonological', 'featural']
            CONTEXT_FEATURES = get_context(ctx, task)
            tgt_si_len = len(SIDE_FEATURES)
            src_si_len = len(SIDE_FEATURES)
            lang_pair_si_len = len(CONTEXT_FEATURES)
            reg_x = params.reg_x
            reg_y = params.reg_y
            reg_z = params.reg_z
            SIDE_INFO_DICT = get_language_pair_side_info(DIR, CONTEXT_FEATURES, SRC, TGT)
            SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = get_language_side_information(DIR, SRC, TGT)


        elif attributes == 'ctx':
            SIDE_FEATURES = None
            CONTEXT_FEATURES = get_context(ctx, task)
            lang_pair_si_len = len(CONTEXT_FEATURES)
            tgt_si_len = 0
            src_si_len = 0
            reg_x = [None]
            reg_y = [None]
            reg_z = params.reg_z
            SIDE_INFO_DICT = get_language_pair_side_info(DIR, CONTEXT_FEATURES, SRC, TGT)
            SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT = None, None


    reg_w = params.reg_w
    reg_h = params.reg_h
    reg_bias_s = params.reg_bias_s
    reg_bias_t = params.reg_bias_t
    BETA = [reg_w, reg_h, reg_x, reg_y, reg_z, reg_bias_s, reg_bias_t]

    return SRC, TGT, SCORE, SIDE_FEATURES, CONTEXT_FEATURES, tgt_si_len, src_si_len, \
           lang_pair_si_len, BETA, SIDE_INFO_DICT, SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT, ctx, attributes


def get_context(context, task):

    if task == 'wiki_aaMF':
        ctx = ['dataset size (sent)', 'Source lang word TTR', 'Source lang subword TTR', 'Target lang word TTR',
               'Target lang subword TTR', 'Source lang vocab size', 'Source lang subword vocab size',
               'Target lang subword vocab size', 'Target lang vocab size', 'Source lang Average Sent. Length',
               'Target lang average sent. length', 'Source lang word Count', 'Target lang word Count',
               'Source lang subword Count', 'Target lang subword Count', 'geographic', 'genetic', 'inventory',
               'syntactic', 'phonological', 'featural']

    elif task == 'bli_aaMF':
        ctx = ['geographic', 'genetic', 'inventory', 'syntactic', 'phonological', 'featural']

    elif task == 'tsfpos_aaMF':
        ctx = ['Rank', 'Accuracy level', 'Overlap word-level', 'Transfer lang dataset size', 'Target lang dataset size', 'Transfer over target size ratio',
               'Transfer lang TTR', 'Target lang TTR', 'Transfer target TTR distance', 'GENETIC', 'SYNTACTIC', 'FEATURAL', 'PHONOLOGICAL', 'INVENTORY', 'GEOGRAPHIC']

    elif task == 'tsfparsing_aaMF':
        ctx = ['Rank', 'Accuracy level', 'Word overlap', 'Transfer lang dataset size', 'Target lang dataset size', 'Transfer over target size ratio',
               'Transfer lang TTR', 'Target lang TTR', 'Transfer target TTR distance', 'GENETIC', 'SYNTACTIC', 'FEATURAL', 'PHONOLOGICAL', 'INVENTORY', 'GEOGRAPHIC' ]

    elif task == 'tsfmt_aaMF':
        ctx = ['Rank', 'BLEU level', 'Overlap word-level', 'Overlap subword-level', 'Transfer lang dataset size', 'Target lang dataset size',
               'Transfer over target size ratio', 'Transfer lang TTR', 'Target lang TTR', 'Transfer target TTR distance', 'GENETIC', 'SYNTACTIC',
               'FEATURAL', 'PHONOLOGICAL', 'INVENTORY',	'GEOGRAPHIC', 'GENETIC_2', 'SYNTACTIC_2', 'FEATURAL_2', 'PHONOLOGICAL_2', 'INVENTORY_2', 'GEOGRAPHIC_2']

    elif task == 'tsfel_aaMF':
        ctx = ['Rank', 'Accuracy level', 'Entity overlap', 'Transfer lang dataset size', 'Target lang dataset size', 'Transfer over target size ratio', 'GENETIC',
            'SYNTACTIC', 'FEATURAL', 'PHONOLOGICAL', 'INVENTORY', 'GEOGRAPHIC']

    return ctx


def get_filename(params, ctx):
    if params.task == 'wiki_MF' or params.task == 'bli_MF':
        filename = f'task_{params.task}_dim_{params.dim}_runs_{params.runs}_it_{params.iterations}_folds_{params.folds}_split_{params.split}'
    else:
        filename = f'task_{params.task}_dim_{params.dim}_attr_{params.attribute}_ctxno_{params.context}_runs_{params.runs}_it_{params.iterations}_folds_{params.folds}_split_{params.split}'

    return filename
