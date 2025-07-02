import tarfile
import shutil
from pathlib import Path
import logging
from typing import Optional

def tar_directory(source_dir: Path, archive_name: Optional[str] = None, 
                 remove_original: bool = True, logger: Optional[logging.Logger] = None) -> bool:
    """
    Create a tar archive of a directory and optionally remove the original.
    
    Args:
        source_dir: Directory to archive
        archive_name: Name for the archive (defaults to source_dir.name + '.tar.gz')
        remove_original: Whether to remove the original directory after archiving
        logger: Logger instance for recording operations
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not source_dir.exists():
        if logger:
            logger.warning(f"Directory to archive does not exist: {source_dir}")
        return False
    
    if archive_name is None:
        archive_name = f"{source_dir.name}.tar.gz"
    
    archive_path = source_dir.parent / archive_name
    
    try:
        if logger:
            logger.info(f"Creating archive: {archive_path}")
        
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(source_dir, arcname=source_dir.name)
        
        # Verify archive was created successfully
        if not archive_path.exists():
            if logger:
                logger.error(f"Archive creation failed: {archive_path}")
            return False
        
        # Remove original directory if requested
        if remove_original:
            if logger:
                logger.info(f"Removing original directory: {source_dir}")
            shutil.rmtree(source_dir)
        
        if logger:
            logger.info(f"Successfully archived {source_dir} to {archive_path}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating archive {archive_path}: {str(e)}")
        return False