import os
import glob


class Loader:
    """Parent class to load datafrom implimented datasets. To build ground truth datasets."""

    FOLDERS = None
    DATASET = None

    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        self.img_file_names = []
        self.tmp_img_file_names = []
        self.mask_file_names = []
        self.data_path = data_path
        self.gt_standard = gt_standard # Get ground truth standard
        self.gt_type = gt_type # Get the ground truth type
        self.ext = ext

        assert (self.gt_standard == 'GT' or self.gt_standard == 'ST'), f"Illegal ground truth standard '{self.gt_standard}' \
            only 'GT' or 'ST' allowed"

    def _get_unique_folders(self):
        """get unique folders and for specific dataset"""
        if self.FOLDERS and self.DATASET:
            for folder in self.FOLDERS:
                self.tmp_img_file_names += glob.glob(f"{self.data_path}/{self.DATASET}/{folder}/*.tif")
                if not self.tmp_img_file_names:
                    return False
            
            self.tmp_img_file_names = sorted(self.tmp_img_file_names)
            # modify each image path for mask path and check if exists. If not pop current index to clean any missing data
            for idx, imgname in enumerate(self.tmp_img_file_names):
                path, basename = os.path.split(imgname)
                nametype = 'man_track' if self.gt_type == 'TRA' else 'man_seg'
                if self.ext != '.tif':
                    basename = basename[1:].replace('.tif', f'{self.ext}')
                else:
                    basename = basename[1:]
                target = f"{path}_{self.gt_standard}/{self.gt_type}/{nametype}{basename}"
                if os.path.exists(imgname) and os.path.exists(target):
                    self.mask_file_names += [target]
                    self.img_file_names += [imgname]
                
            # check img and mask files are the same length
            assert len(self.img_file_names) == len(self.mask_file_names), 'Image files do not match the number of mask files'
            assert len(self.img_file_names) != 0, f'Warning no files loaded. Check gt_type {self.gt_type}, and file extension. Using {self.ext}'
            return True
        else:
            return False

    def return_img_mask_files(self):
        """retrun image and mask files for dataset"""
        return self.img_file_names, self.mask_file_names
        
class Loader_BFC2DLHSC(Loader):
    """
    Mouse hematopoietic stem cells in hydrogel microwells

    Dr. H. Blau, Baxter Laboratory for Stem Cell Biology, Stanford University, USA

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip✱ (1.6 GB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/BF-C2DL-HSC.zip (1.6 GB)
    Less details

    Microscope: Zeiss PALM/AxioObserver Z1

    Objective lens: EC Plan-Neofluar 10x/0.30 Ph1

    Pixel size (microns): 0.645 x 0.645

    Time step (min): 5
    
    """
    FOLDERS = ['01', '02']
    DATASET = 'BF-C2DL-HSC'
    STANDARD = ['GT', 'ST']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'

   
class Loader_BFC2DLMuSC(Loader):
    """
    Mouse muscle stem cells in hydrogel microwells

    Dr. H. Blau, Baxter Laboratory for Stem Cell Biology, Stanford University, USA

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-MuSC.zip✱ (1.2 GB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/BF-C2DL-MuSC.zip (1.3 GB)
    Less details

    Microscope: Zeiss PALM/AxioObserver Z1

    Objective lens: EC Plan-Neofluar 10x/0.30 Ph1

    Pixel size (microns): 0.645 x 0.645

    Time step (min): 5
    """
    FOLDERS = ['01', '02']
    DATASET = 'BF-C2DL-MuSC'
    STANDARD = ['GT', 'ST']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'

class Loader_DICC2DHHeLa(Loader):
    """
    HeLa cells on a flat glass

    Dr. G. van Cappellen. Erasmus Medical Center, Rotterdam, The Netherlands

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip✱ (37 MB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/DIC-C2DH-HeLa.zip (41 MB)
    Less details

    Microscope: Zeiss LSM 510 Meta

    Objective lens: Plan-Apochromat 63x/1.4 (oil)

    Pixel size (microns): 0.19 x 0.19

    Time step (min): 10
    
    """

    FOLDERS = ['01', '02']
    DATASET = 'DIC-C2DH-HeLa'
    STANDARD = ['GT', 'ST']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'


class Loader_FluoC2DLHuh7(Loader):
    """
    Human hepatocarcinoma-derived cells expressing the fusion protein YFP-TIA-1

    Dr. Alessia Ruggieri and Philipp Klein, Centre for Integrative Infectious Disease Research (CIID), University Hospital Heidelberg, Germany

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip (36 MB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C2DL-Huh7.zip (36 MB)
    Less details

    Microscope: Nikon Eclipse Ti2

    Objective lens: CFI Plan Apo Lambda 20x/0.75

    Pixel size (microns): 0.65 x 0.65

    Time step (min): 15

    Additional information: Cell Host & Microbe, 2012
    """

    FOLDERS = ['01', '02']
    DATASET = 'Fluo-C2DL-Huh7'
    STANDARD = ['GT']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'
     
        
class Loader_FluoC2DLMSC(Loader):

    """
    Rat mesenchymal stem cells on a flat polyacrylamide substrate

    Dr. F. Prósper. Cell Therapy laboratory, Center for Applied Medical Research (CIMA), Pamplona, Spain

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip✱ (72 MB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C2DL-MSC.zip (71 MB)
    Less details

    Microscope: PerkinElmer UltraVIEW ERS

    Objective lens: Plan-Neofluar 10x/0.3 (Plan-Apo 20x/0.75)

    Pixel size (microns): 0.3 x 0.3 (0.3977 x 0.3977)

    Time step (min): 20 (30)
    """

    FOLDERS = ['01', '02']
    DATASET = 'Fluo-C2DL-MSC'
    STANDARD = ['GT', 'ST']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'
        


class Loader_FluoN2DHGOWT1(Loader):

    """
    GFP-GOWT1 mouse stem cells

    Dr. E. Bártová. Institute of Biophysics, Academy of Sciences of the Czech Republic, Brno, Czech Republic

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip✱ (53 MB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DH-GOWT1.zip (46 MB)
    Less details

    Microscope: Leica TCS SP5

    Objective lens: Plan-Apochromat 63x/1.4 (oil)

    Pixel size (microns): 0.240 x 0.240

    Time step (min): 5

    Additional information: PLoS ONE, 2011

    """

    FOLDERS = ['01', '02']
    DATASET = 'Fluo-N2DH-GOWT1'
    STANDARD = ['GT', 'ST']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'
        
class Loader_FluoN2DHGOWT1(Loader):

    """
    GFP-GOWT1 mouse stem cells

    Dr. E. Bártová. Institute of Biophysics, Academy of Sciences of the Czech Republic, Brno, Czech Republic

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip✱ (53 MB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DH-GOWT1.zip (46 MB)
    Less details

    Microscope: Leica TCS SP5

    Objective lens: Plan-Apochromat 63x/1.4 (oil)

    Pixel size (microns): 0.240 x 0.240

    Time step (min): 5

    Additional information: PLoS ONE, 2011

    """

    FOLDERS = ['01', '02']
    DATASET = 'Fluo-N2DH-GOWT1'
    STANDARD = ['GT', 'ST']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'
        

class Loader_FluoN2DLHeLa(Loader):
    """
    HeLa cells stably expressing H2b-GFP

    Mitocheck Consortium

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip✱ (182 MB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DL-HeLa.zip (168 MB)
    Less details

    Microscope: Olympus IX81

    Objective lens: Plan 10x/0.4

    Pixel size (microns): 0.645 x 0.645

    Time step (min): 30

    Additional information: Nature, 2010
    """

    FOLDERS = ['01', '02']
    DATASET = 'Fluo-N2DL-HeLa'
    STANDARD = ['GT', 'ST']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'
        
class Loader_PhCC2DHU373(Loader):

    """
    Glioblastoma-astrocytoma U373 cells on a polyacrylamide substrate

    Dr. S. Kumar. Department of Bioengineering, University of California at Berkeley, Berkeley CA (USA)

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip✱ (40 MB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/PhC-C2DH-U373.zip (38 MB)
    Less details

    Microscope: Nikon

    Objective lens: Plan Fluor DLL 20x/0.5

    Pixel size (microns): 0.65 x 0.65

    Time step (min): 15
    """

    FOLDERS = ['01', '02']
    DATASET = 'PhC-C2DH-U373'
    STANDARD = ['GT', 'ST']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'
        

class Loader_PhCC2DLPSC(Loader):
    """
    Pancreatic stem cells on a polystyrene substrate

    Dr. T. Becker and Dr. D. Rapoport. Fraunhofer Institution for Marine Biotechnology, Lübeck, Germany

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip✱ (124 MB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/PhC-C2DL-PSC.zip (106 MB)
    Less details

    Microscope: Olympus ix-81

    Objective lens: UPLFLN 4XPH

    Pixel size (microns): 1.6 x 1.6

    Time step (min): 10

    Additional information: PLoS ONE, 2011
    """

    FOLDERS = ['01', '02']
    DATASET = 'PhC-C2DL-PSC'
    STANDARD = ['GT', 'ST']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'
        
class Loader_FluoN2DHSIM(Loader):
    """
    Simulated nuclei of HL60 cells stained with Hoescht

    Dr. V. Ulman and Dr. D. Svoboda. Centre for Biomedical Image Analysis (CBIA),

    Masaryk University, Brno, Czech Republic (Created using MitoGen, part of Cytopacq)

    Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip (91 MB)

    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DH-SIM+.zip (96 MB)
    Less details

    Microscope: Zeiss Axiovert 100S with a Micromax 1300-YHS camera

    Objective lens: Plan-Apochromat 40x/1.3 (oil)

    Pixel size (microns): 0.125 x 0.125

    Time step (min): 29

    Additional information: IEEE Transactions on Medical Imaging, 2016
    """

    FOLDERS = ['01', '02']
    DATASET = 'Fluo-N2DH-SIM'
    STANDARD = ['GT']
    
    def __init__(self, data_path, gt_standard, gt_type, ext='.tif'):
        Loader.__init__(self, data_path, gt_standard, gt_type, ext)
        load = self._get_unique_folders()
        assert load, f'Warning no images found check {data_path}'
        


DATASETS = {
        'BF-C2DL-HSC': Loader_BFC2DLHSC, 
        'BF-C2DL-MuSC':Loader_BFC2DLMuSC, 
        'DIC-C2DH-HeLa':Loader_DICC2DHHeLa, 
        'Fluo-C2DL-Huh7':Loader_FluoC2DLHuh7, 
        'Fluo-C2DL-MSC':Loader_FluoC2DLMSC, 
        'Fluo-N2DH-GOWT1':Loader_FluoN2DHGOWT1,
        'Fluo-N2DL-HeLa': Loader_FluoN2DLHeLa,
        'PhC-C2DH-U373': Loader_PhCC2DHU373,
        'PhC-C2DL-PSC': Loader_PhCC2DLPSC,
        'Fluo-N2DH-SIM': Loader_FluoN2DHSIM,
    }



