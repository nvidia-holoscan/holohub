# Copyright 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Union

from .dicom_series import DICOMSeries
from .dicom_study import DICOMStudy
from .domain import Domain
from .image import Image


class SelectedSeries(Domain):
    """This class encapsulates a DICOM series that has been selected with a given selection name.

    It references the DICOMSeries object with the name associated with selection, and a Image if applicable.
    """

    def __init__(
        self,
        selection_name: str,
        dicom_series: DICOMSeries,
        image: Optional[Image] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Creates an instance of this class

        Args:
            selection_name (str): Name given to this selection, if None, then series instance UID is used.
            dicom_series (DICOMSeries): The referenced DICOMSeries object.
            image (Optional[Image], optional): Converted image of the series, if applicable. Defaults to None.
            metadata (Optional[Dict], optional): Metadata dictionary for the instance. Defaults to None.

        Raises:
            ValueError: If argument dicom_series is not a DICOMSeries object.
        """
        if not isinstance(dicom_series, DICOMSeries):
            raise ValueError("Argument dicom_series must be a DICOMSeries object.")
        super().__init__(metadata)
        self._series = dicom_series

        if not image:
            self._image = None
        elif isinstance(image, Image):
            self._image = image
        else:
            raise ValueError("'image' must be an Image object if not None.")

        if not selection_name or not selection_name.strip():
            self._name = str(dicom_series.SeriesInstanceUID)
        else:
            self._name = selection_name.strip()

    @property
    def series(self) -> DICOMSeries:
        return self._series

    @property
    def selection_name(self) -> str:
        return self._name

    @property
    def image(self) -> Union[Image, None]:
        return self._image

    @image.setter
    def image(self, val: Optional[Image]):
        if not val:
            self._image = None
        elif isinstance(val, Image):
            self._image = val
        else:
            raise ValueError("'val' must be an Image object if not None.")


class StudySelectedSeries(Domain):
    """This class encapsulates a DICOM study and a list of selected DICOM series within it.

    It references the DICOMStudy object and a dictionary of SelectedSeries objects.
    """

    def __init__(self, study: DICOMStudy, metadata: Optional[Dict] = None) -> None:
        """Creates a instance with a DICOMStudy object.

        Args:
            study (DICOMStudy): The DICOMStudy object referenced.
            metadata (Optional[Dict], optional): Metadata dictionary for the instance. Defaults to None.

        Raises:
            ValueError: If argument study is not a DICOMStudy object.
        """
        if not isinstance(study, DICOMStudy):
            raise ValueError("A DICOMStudy object is required.")
        super().__init__(metadata)
        self._study: DICOMStudy = study
        self._select_series_dict: Dict = {}  # "selection_name": [SelectedSeries]

    @property
    def study(self) -> DICOMStudy:
        """Returns the DICOMStudy object referenced.

        Returns:
            DICOMStudy: The referenced DICOMStudy object.
        """
        return self._study

    @property
    def selected_series(self) -> List[SelectedSeries]:
        """Returns a view of the list of all the SelectedSeries objects.

        Returns:
            List[SelectedSeries]: A view of the flat list of all SelectedSeries objects.
        """
        list_of_sublists = list(self._select_series_dict.values())
        return [item for sublist in list_of_sublists for item in sublist]

    @property
    def series_by_selection_name(self) -> Dict:
        """Returns the list of SelectedSeries by selection names in a dictionary.

        Returns:
            Dict: Dictionary with selection name as key and list of SelectedSeries as value.
        """
        return self._select_series_dict

    def add_selected_series(self, selected_series: SelectedSeries) -> None:
        """Adds a SelectedSeries object in the referenced DICOMStudy.

        The SelectedSeries object is grouped by the selection name in a list,
        so there could be one or more objects for a given selection name.

        Args:
            selected_series (SelectedSeries): The reference of the SelectedSeries object.

        Raises:
            ValueError: If argument selected_series is not a SelectedSeries object.
        """
        if not isinstance(selected_series, SelectedSeries):
            raise ValueError("A SelectedSeries object is required.")
        selected_series_list = self._select_series_dict.get(selected_series.selection_name, [])
        selected_series_list.append(selected_series)
        self._select_series_dict[selected_series.selection_name] = selected_series_list
