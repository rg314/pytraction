
import os
from google_drive_downloader import GoogleDriveDownloader as gdd



def main():
    """get example data from Google Drive
    """
    # data_20210320.zip
    file_id = '1DsPuqAzI7CEH-0QN-DWHdnF6-to5HdFe'
    destination = 'data/data.zip'

    if not os.path.exists('data/'):
        os.mkdir('data')


    gdd.download_file_from_google_drive(file_id=file_id,
                                    dest_path=destination,
                                    unzip=True,
                                    showsize=True,
                                    overwrite=False)
    return True