from . import matchers
from . import extractors

def load_component(compo_name,model_name,config):
    if compo_name=='extractor':
        component=load_extractor(model_name,config)
    elif compo_name=='matcher':
        component=load_matcher(model_name,config)
    else:
        raise NotImplementedError
    return component


def load_extractor(model_name,config):
    if model_name=='sp_light':
        print('Using Superpoint extractor')   
        extractor=extractors.ExtractSuperpoint_Light(config)
    elif model_name=='sp_sift':
        print('Using SuperSift extractor')
        extractor=extractors.ExtractSuperSift(config)
    else:
        raise NotImplementedError
    return extractor

def load_matcher(model_name,config):
    if model_name=='sp_sift_sg':
        print('Matching with SuperSift SuperGlue matcher')
        matcher=matchers.GNN_Matcher(config,'sp_sift_sg')
    elif model_name=='SG':
        print('Matching with SuperGlue matcher')
        matcher=matchers.GNN_Matcher(config,'SG')
    else:
        raise NotImplementedError
    return matcher
