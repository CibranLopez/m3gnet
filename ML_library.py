import numpy  as np
import pandas as pd

from os                       import path, listdir
from sklearn.model_selection  import train_test_split
from pymatgen.io.vasp.outputs import Vasprun, Outcar

def is_relaxation_folder_valid(path_to_relaxation):
    """Determines whether path_to_relaxation contains a vasprun.xml file or not.
    It returns True (valid) if the file does exist.
    """
    
    if path.exists(f'{path_to_relaxation}/vasprun.xml'):
        return True
    return False


def clean_vasprun(path_to_relaxation):
    """Rewrite those lines from the vasprun like '<field>  </field>', as somentimes invalid characters appear.
    
    Args:
        path_to_relaxation (str): path to the folder which contains the vasprun.xml file.
    """
    
    # Load lines
    with open(f'{path_to_relaxation}/vasprun.xml', 'r') as file:
        lines = file.readlines()

    # Rewrite them, avoiding invalid characters
    with open(f'{path_to_relaxation}/vasprun.xml', 'w') as file:
        for line in lines:
            split_line = line.split()
            if split_line[0][:7] == '<field>':
                    if (len(split_line) == 2) and (split_line[0][1][:7]):
                        file.write('     <field>  </field>\n')
                    else:
                        file.write(line)
            else:
                file.write(line)


def split_data(data, test_size=0.2, validation_size=0.2, random_state=None):
    """
    Split a Pandas DataFrame into training, validation, and test sets.

    Parameters:
        data (DataFrame): The input dataset to be split.
        test_size (float): The proportion of data to include in the test set (default: 0.2).
        validation_size (float): The proportion of data to include in the validation set (default: 0.2).
        random_state (int or None): Seed for the random number generator (default: None).

    Returns:
        train_set (DataFrame): The training dataset.
        validation_set (DataFrame): The validation dataset.
        test_set (DataFrame): The test dataset.
    """
    
    # First, split the data into training and temporary data (temporary_data = validation + test)
    train_data, temp_data = train_test_split(data, test_size=(validation_size + test_size), random_state=random_state)

    # Next, split the temporary data into validation and test data
    validation_data, test_data = train_test_split(temp_data, test_size=(test_size / (validation_size + test_size)), random_state=random_state)

    return train_data, validation_data, test_data


def extract_vaspruns_dataset(path_to_dataset):
    """Generates a Pandas DataFrame with the data from each simulation in the path. It assumes the following disposition:
    
    Theory level
        Material
            Relaxation step
                Defect state
                    vasprun.xml
    
    Args:
        path_to_dataset (str): Path to tree database containing different level of theory calculations.
    
    Returns:
        m3gnet_dataset (Pandas DataFrame): DataFrame with information of simulations in multicolumn format (material, defect state, ionic step).
    """
    
    # Initialize the data dictionary
    data = {}
    
    # Initialize dataset with MP format
    columns = ['structure', 'energy', 'force', 'stress']
    
    # Iterate over materials and relaxations in the dataset
    for material in listdir(path_to_dataset):
        # Define path to material
        path_to_material = f'{path_to_dataset}/{material}'

        # Check if it is a folder
        if not path.isdir(path_to_material):
            continue

        print()
        print(material)
        
        # Get relaxations steps (rel1, rel2...)
        relaxation_steps = listdir(path_to_material)
        
        # Determine all defect states across every folder
        defect_states = []
        for relaxation_step in relaxation_steps:
            path_to_relaxation_step = f'{path_to_material}/{relaxation_step}'
            if path.isdir(path_to_relaxation_step):
                for defect_state in listdir(path_to_relaxation_step):
                    if path.isdir(f'{path_to_material}/{relaxation_step}/{defect_state}'):
                        defect_states.append(defect_state)
        
        # Determine unique defect states across every folder
        unique_defect_states = np.unique(defect_states)
        
        # Run over all defect states
        for defect_state in unique_defect_states:
            print(f'\t{defect_state}')
            
            # Run over all relaxation steps
            for relaxation_step in relaxation_steps:
                # Define path to relaxation loading every relaxation step of a same defect state in the same data column
                path_to_relaxation = f'{path_to_material}/{relaxation_step}/{defect_state}'
                
                # Check if it is a valid relaxation (with a vasprun.xml file)
                if not is_relaxation_folder_valid(path_to_relaxation):
                    continue
                
                temp_relaxation = f'{material}_{defect_state}'
                
                # Remove invalid characters from the vasprun.xml file
                clean_vasprun(path_to_relaxation)  # Uncomment is it happens to you as well!!
                
                # Load data from relaxation
                try:
                    # Try to load those unfinished relaxations as well
                    vasprun = Vasprun(f'{path_to_relaxation}/vasprun.xml', exception_on_bad_xml=False)
                except:
                    print('Error: vasprun not correctly loaded.')
                    continue
                
                # Run over ionic steps
                for ionic_step_idx in range(len(vasprun.ionic_steps)):
                    temp_ionic_step = f'{temp_relaxation}_{ionic_step_idx}'
                    # Extract data from each ionic step
                    temp_structure = vasprun.ionic_steps[ionic_step_idx]['structure']
                    temp_energy    = vasprun.ionic_steps[ionic_step_idx]['e_fr_energy']
                    temp_forces    = vasprun.ionic_steps[ionic_step_idx]['forces']
                    temp_stress    = vasprun.ionic_steps[ionic_step_idx]['stress']
                    
                    # Stresses obtained from VASP calculations (default unit is kBar) should be multiplied by -0.1
                    # to work directly with the model
                    temp_stress = np.array(temp_stress)
                    temp_stress *= -0.1
                    temp_stress = temp_stress.tolist()
                    
                    # Generate a dictionary object with the new data
                    new_data = {(material, temp_relaxation, temp_ionic_step): [temp_structure, temp_energy, temp_forces, temp_stress]}
                    
                    # Update in the main data object
                    data.update(new_data)

    # Convert to Pandas DataFrame
    m3gnet_dataset = pd.DataFrame(data, index=columns)
    return m3gnet_dataset
