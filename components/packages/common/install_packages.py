import os

import logger


def install_requirements(requirements):
    pypi_mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
    for package_name, package_version in requirements.items():
        if package_version:
            cmd = f"pip install -i {pypi_mirror} {package_name}=={package_version}"  # noqa
        else:
            cmd = f"pip install -i {pypi_mirror} {package_name}"
        error = os.system(cmd)
        if not error:
            logger.info(f"Install package {package_name} with version {package_version} successfully.")  # noqa
        else:
            logger.error(f"Install package {package_name} with version {package_version} failed!")  # noqa
