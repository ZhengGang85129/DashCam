#!/usr/bin/env python3
import yaml

ALLOWED_AUTMENTATION_TYPES = [ "fog", "noise", "gaussian_blur", "color_jitter", "horizontal_flip", "rain_effect" ]
ALLOWED_MODEL_TYPES = ['timesformer', 'baseline', 'accidentxai', 'swintransformer', "mvit_v2"]

class Args:
    def __init__(self, confDICT):
        self.num_workers            =   int(confDICT.get('num_workers', 4))
        self.augmentation_types     =       confDICT.get('augmentation_types', None)
        self.augmentation_prob      = float(confDICT.get('augmentation_prob', 0.))
        self.horizontal_flip_prob   = float(confDICT.get('horizontal_flip_prob', 0.))
        self.model_type             =   str(confDICT.get('model_type', 'baseline'))
        #self.optimizer              =   str(confDICT.get('optimizer', 'radam'))

        #self.model_dir              =       confDICT.get('model_dir', 'model_ckpt')
        self.debug                  =  bool(confDICT['debug']) if 'debug' in confDICT else False
        self.print_freq = int(confDICT.get('print_freq', 4))
        
        self.training_dir = str(confDICT.get('training_dir', None))
        self.validation_dir = str(confDICT.get('validation_dir', None))
        self.test_dir = str(confDICT.get('test_dir', None))
        self.test_csv = str(confDICT.get('test_csv', None))
        self.training_csv = str(confDICT.get('training_csv', None))
        self.validation_csv = str(confDICT.get('validation_csv', None))
        
        
        self.training_strategy = confDICT.get('training_strategy', None)
        self.classifier = confDICT.get('classifier', None) 
        
        self.num_train_samples = confDICT.get('num_train_samples', 6000)
        self.num_val_samples = confDICT.get('num_val_samples', 1200)
        self.seed = confDICT.get('SEED', 123) 
        
        self.total_epochs = confDICT.get('total_epochs', 10)
        if isinstance(self.total_epochs, str):
            self.total_epochs = int(self.total_epochs)
        
        self.scheduler = confDICT.get('scheduler', None)
        self.scheduler['T_max'] =  self.total_epochs
        
         
        if self.augmentation_types is None:
            pass ### if nothing set. use default value
        else:
            if not isinstance(self.augmentation_types, list):
                raise IOError(f'[InvalidArgumentType] "augmentation_types" required a list but type "{ type(self.augmentation_types) }" in yaml file')
            if len(self.augmentation_types) == 0:
                raise IOError(f'[ArgumentRequired] "augmentation_types" not found in Yaml file. It is required. The available options are "{ALLOWED_AUTMENTATION_TYPES}"')
            for aug_type in self.augmentation_types:
                if aug_type not in ALLOWED_AUTMENTATION_TYPES:
                    raise IOError(f'[InvalidAugmentationType] "{ aug_type }" is not allowed. The allowed values: "{ALLOWED_AUTMENTATION_TYPES}"')

        if self.model_type not in ALLOWED_MODEL_TYPES:
            raise IOError(f'[InvalidModelType] "{ self.model_type }" is not allowed. The allowed values: "{ ALLOWED_MODEL_TYPES }"')


def yaml_content_args(yamlCONTENT) -> Args:
    return Args( yaml.safe_load(yamlCONTENT) )

def load_yaml_file_from_arg(yamlFILE:str):
    with open(yamlFILE,'r') as fIN:
        return yaml_content_args(fIN)



def test_yaml_args() -> dict:
    yaml_content = '''
monitor_dir: monitor_train
learning_rate: 0.0001
epochs: 20
batch_size: 10
num_workers: 4
augmentation_types:
    - fog
    - noise
    - gaussian_blur
    - color_jitter
    - rain_effect
augmentation_prob: 0.25
horizontal_flip_prob: 0.5

model_dir: model_ckpt
debug: False
    '''
    print(yaml_content_args(yaml_content))

def validate_yaml_files():
    '''
    Validate the input yaml file before submit the condor job
    '''
    import sys
    if len(sys.argv) < 1+1:
        print(f'[InputYamlFiles] Please put yaml files in the argument. Usage: python3 thisfile.py conf1.yaml conf2.yaml')

    for yamlFILE in sys.argv[1:]:
        print(f'[ReadingYamlFile] {yamlFILE}')
        load_yaml_file_from_arg(yamlFILE)
        print(f'[GOOOOOOOOOOOOOD] verified')

if __name__ == "__main__":
    #test_yaml_args()
    #return print(load_yaml_file_from_arg('train.args.yaml').debug)
    validate_yaml_files()

