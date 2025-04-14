import numpy as np


class POD:
    """This class contains all algorithms for the proper orthogonal
       decomposition (POD)."""
       
    POD_snapshots=[]
    basis_fts_matrix=[]
    information_content=[]
    
    
    def POD_parameters(self, threshold_percent=99.999):
        """ Performs the POD for the parameters. This POD algorithm should be
            used for time-independent (steady-state) applications

        Args:
        n_variables = number of variables for which the POD needs to be performed
        threshold_percent = vector of dimension of the number of variables that
                            defines the desired accuracy

        Returns:

        """
        svd = []
        # Perform the Singular Value Decomposition
        svd.append(np.linalg.svd(self.Pod_snapshots[0,:,:],  full_matrices=False)) 
        
        print()
    
        sum_eigen_full = np.sum(svd[0][1]**2)
        index=0
            
        print('Selecting the basis functions for the reduced basis')
        
        for i in range(len(svd[0][1])):
            sum_eigen = np.sum(svd[0][1][:i]**2)
            if((sum_eigen/sum_eigen_full)<=(threshold_percent/100)):
               self.information_content.append(sum_eigen/sum_eigen_full)
               index+=1
        self.basis_fts_matrix.append(np.copy(svd[0][2][:index,:]))
