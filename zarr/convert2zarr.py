import os
import sys
import zarr
import dxchange
import yaml
import pickle
import pathlib
import meta

def save_pickle(file_name, parameters):
    """Pickle saves and load exactly as it was, but is not human readable."""
    with open(file_name, 'wb') as f:
        pickle.dump(parameters, f)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        loaded_parameters = pickle.load(f)
    return loaded_parameters

def save_yaml(file_name, parameters):
    """YAML Ain't Markup Language

    Pros
    ----
    - Keys do not need to be strings.
    - The syntax is also less verbose because it doesn't use brackets.

    Gotchas
    -------
    - YAML is not part of the standard library.
    - Syntax for storing NumPy arrays is verbose!
    """

    with open(file_name, 'w') as f:
        # Can use yaml.CDumper for faster dump/load if available
        yaml.dump(parameters, f, indent=4, Dumper=yaml.Dumper)

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        loaded_parameters = yaml.load(f, Loader=yaml.Loader)

    return loaded_parameters

def dict_compare(parameters, loaded_parameters):

    for key, value in loaded_parameters.items():
        if loaded_parameters[key] == parameters[key]:
            print('ok', key, value)
        else:
            print('different', key, loaded_parameters[key][0], parameters[key][0])
    if parameters == loaded_parameters:
        print('True')
    else:
        print('False')

def main(args):

    if len(sys.argv) == 1:
        print ('ERROR: Must provide the path to a run-file folder as the argument')
        print ('Example:')
        print ('        python %s /data/file.h5'% sys.argv[0])
        sys.exit(1)
    else:

        file_name   = sys.argv[1]
        p = pathlib.Path(file_name)
        if p.is_file():
            p = pathlib.Path(file_name).joinpath(p.stem)
            file_name_hdf   = p.with_suffix('.h5')
            file_name_pickle   = p.with_suffix('.pickle')
            file_name_yaml   = p.with_suffix('.yaml')
            # file_name_zarr_data = p.with_suffix('.dtxml')
            # file_name_zarr_flat  = p.parents[1].joinpath('master').with_suffix('.xlsx')
        else:
            print('ERROR: %s does not exist' % p)
            sys.exit(1)
    
    base_name    = os.path.splitext(file_name)[0] 

    data, flat, dark, theta = dxchange.read_aps_tomoscan_hdf5(file_name)#, sino=(100, 400))
    
    zarr.save(base_name + '/data.zarr', data)
    zarr.save(base_name + '/flat.zarr', flat)
    zarr.save(base_name + '/dark.zarr', dark)

    # _, meta_dict = dxchange.read_hdf_meta(file_name) # this uses the meta data reader from dxchange
    mp = meta.read_meta.Hdf5MetadataReader(file_name)
    meta_dict = mp.readMetadata()
    mp.close()
    
    file_name_yaml   = base_name + '/meta_data.yaml'
    save_yaml(file_name_yaml, meta_dict)

    file_name_pickle = base_name + '/meta_data.pickle'
    save_pickle(file_name_pickle, meta_dict)

if __name__ == '__main__':
    main(sys.argv)
