__version__ = '0.1.0a22'

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import logging.config

from molcraft import chem
from molcraft import features
from molcraft import descriptors
from molcraft import featurizers
from molcraft import layers 
from molcraft import models 
from molcraft import ops 
from molcraft import records 
from molcraft import tensors
from molcraft import callbacks
from molcraft import datasets
from molcraft import losses


logging.config.dictConfig(
    { 
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': { 
            'standard': { 
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': { 
            'default': { 
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
        },
        'loggers': { 
            '': {
                'handlers': ['default'],
                'level': 'WARNING',
                'propagate': False
            },
            'molcraft': { 
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': False
            },
            '__main__': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': False
            },
        } 
    }
)

logger = logging.getLogger(__name__)
logger.debug("Logging is configured.")