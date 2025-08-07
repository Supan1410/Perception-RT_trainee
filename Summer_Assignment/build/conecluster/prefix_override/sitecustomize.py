import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/supan/RT/Perception/Summer_Assignment/install/conecluster'
