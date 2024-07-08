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


from .domain import Domain


class DICOMStudy(Domain):
    """This class represents a DICOM Study.

    It contains a collection of DICOM Series.
    """

    def __init__(self, study_instance_uid):
        super().__init__(None)
        self._study_instance_uid = study_instance_uid
        self._series_dict = {}
        # Do not set attributes in advance to save memory

    def get_study_instance_uid(self):
        return self._study_instance_uid

    def add_series(self, series):
        self._series_dict[series.get_series_instance_uid()] = series

    def get_all_series(self):
        return list(self._series_dict.values())

    # Properties named after DICOM Study module attribute keywords
    # There is only one required (Type 1) attrbute for a study:
    #     Keyword: StudyInstanceUID, Tag: (0020,000D)
    #
    @property
    def StudyInstanceUID(self):
        return self._study_instance_uid

    @StudyInstanceUID.setter
    def StudyInstanceUID(self, val):
        self._study_instance_uid = val

    @property
    def StudyID(self):
        return getattr(self, "_study_id", None)

    @StudyID.setter
    def StudyID(self, val):
        self._study_id = val

    @property
    def StudyDate(self):
        return getattr(self, "_study_date", None)

    @StudyDate.setter
    def StudyDate(self, val):
        self._study_date = val

    @property
    def StudyTime(self):
        return getattr(self, "_study_time", None)

    @StudyTime.setter
    def StudyTime(self, val):
        self._study_time = val

    @property
    def StudyDescription(self):
        return getattr(self, "_study_description", None)

    @StudyDescription.setter
    def StudyDescription(self, val):
        self._study_description = val

    @property
    def AccessionNumber(self):
        return getattr(self, "_accession_number", None)

    @AccessionNumber.setter
    def AccessionNumber(self, val):
        self._accession_number = val

    def __str__(self):
        result = "\n---------------" + "\n"

        if self._study_instance_uid is not None:
            study_instance_uid_attr = "Study Instance UID: " + self._study_instance_uid + "\n"
            result += study_instance_uid_attr
        if self.StudyID is not None:
            study_id_attr = "Study ID: " + self.StudyID + "\n"
            result += study_id_attr
        if self.StudyDate is not None:
            study_date_attr = "Study Date: " + self.StudyDate + "\n"
            result += study_date_attr
        if self.StudyTime is not None:
            study_time_attr = "Study Time: " + self.StudyTime + "\n"
            result += study_time_attr
        if self.StudyDescription is not None:
            study_desc_attr = "Study Description: " + self.StudyDescription + "\n"
            result += study_desc_attr
        if self.AccessionNumber is not None:
            accession_num_attr = "Accession Number: " + self.AccessionNumber + "\n"
            result += accession_num_attr

        result += "---------------" + "\n"

        return result
