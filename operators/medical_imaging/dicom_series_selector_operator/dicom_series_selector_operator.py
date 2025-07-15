# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numbers
import re
from json import loads as json_loads
from typing import List

from holoscan.core import ConditionType, Fragment, Operator, OperatorSpec

from operators.medical_imaging.core.domain.dicom_series import DICOMSeries
from operators.medical_imaging.core.domain.dicom_series_selection import (
    SelectedSeries,
    StudySelectedSeries,
)
from operators.medical_imaging.core.domain.dicom_study import DICOMStudy


class DICOMSeriesSelectorOperator(Operator):
    """This operator selects a list of DICOM Series in a DICOM Study for a given set of selection rules.

    Named input:
        dicom_study_list: A list of DICOMStudy objects.

    Named output:
        study_selected_series_list: A list of StudySelectedSeries objects. Downstream receiver optional.

    This class can be considered a base class, and a derived class can override the 'filer' function to with
    custom logics.

    In its default implementation, this class
        1. selects a series or all matched series within the scope of a study in a list of studies
        2. uses rules defined in JSON string, see below for details
        3. supports DICOM Study and Series module attribute matching, including regex for string types
        4. supports multiple named selections, in the scope of each DICOM study
        5. outputs a list of SutdySelectedSeries objects, as well as a flat list of SelectedSeries (to be deprecated)

    The selection rules are defined in JSON,
        1. attribute "selections" value is a list of selections
        2. each selection has a "name", and its "conditions" value is a list of matching criteria
        3. each condition has uses the implicit equal operator, except for regex for str type and union for set type
        4. DICOM attribute keywords are used, and only for those defined as DICOMStudy and DICOMSeries properties

    An example selection rules:
    {
        "selections": [
            {
                "name": "CT Series 1",
                "conditions": {
                    "StudyDescription": "(?i)^Spleen",
                    "Modality": "(?i)CT",
                    "SeriesDescription": "(?i)^No series description|(.*?)"
                }
            },
            {
                "name": "CT Series 2",
                "conditions": {
                    "Modality": "CT",
                    "BodyPartExamined": "Abdomen",
                    "SeriesDescription" : "Not to be matched. For illustration only."
                }
            }
        ]
    }
    """

    def __init__(
        self, fragment: Fragment, *args, rules: str = "", all_matched: bool = False, **kwargs
    ) -> None:
        """Instantiate an instance.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            rules (Text): Selection rules in JSON string.
            all_matched (bool): Gets all matched series in a study. Defaults to False for first match only.
        """

        # rules: Text = "", all_matched: bool = False,

        # Delay loading the rules as JSON string till compute time.
        self._rules_json_str = rules if rules and rules.strip() else None
        self._all_matched = all_matched  # all_matched
        self.input_name_study_list = "dicom_study_list"
        self.output_name_selected_series = "study_selected_series_list"

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_study_list)
        spec.output(self.output_name_selected_series).condition(
            ConditionType.NONE
        )  # Receiver optional

        # Can use the config file to alter the selection rules per app run
        # spec.param("selection_rules")

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator."""

        dicom_study_list = op_input.receive(self.input_name_study_list)
        selection_rules = self._load_rules() if self._rules_json_str else None
        study_selected_series = self.filter(selection_rules, dicom_study_list, self._all_matched)
        op_output.emit(study_selected_series, self.output_name_selected_series)

    def filter(
        self, selection_rules, dicom_study_list, all_matched: bool = False
    ) -> List[StudySelectedSeries]:
        """Selects the series with the given matching rules.

        If rules object is None, all series will be returned with series instance UID as the selection name.

        Simplistic matching is used for demonstration:
            Number: exactly matches
            String: matches case insensitive, if fails then tries RegEx search
            String array matches as subset, case insensitive

        Args:
            selection_rules (object): JSON object containing the matching rules.
            dicom_study_list (list): A list of DICOMStudiy objects.
            all_matched (bool): Gets all matched series in a study. Defaults to False for first match only.

        Returns:
            list: A list of objects of type StudySelectedSeries.

        Raises:
            ValueError: If the selection_rules object does not contain "selections" attribute.
        """

        if not dicom_study_list or len(dicom_study_list) < 1:
            return []

        if not selection_rules:
            # Return all series if no selection rules are supplied
            logging.warn("No selection rules given; select all series.")
            return self._select_all_series(dicom_study_list)

        selections = selection_rules.get("selections", None)  # TODO type is not json now.
        # If missing selections in the rules then it is an error.
        if not selections:
            raise ValueError('Expected "selections" not found in the rules.')

        study_selected_series_list = []  # List of StudySelectedSeries objects

        for study in dicom_study_list:
            study_selected_series = StudySelectedSeries(study)
            for selection in selections:
                # Get the selection name. Blank name will be handled by the SelectedSeries
                selection_name = selection.get("name", "").strip()
                logging.info(f"Finding series for Selection named: {selection_name}")

                # Skip if no selection conditions are provided.
                conditions = selection.get("conditions", None)
                if not conditions:
                    continue

                # Select only the first series that matches the conditions, list of one
                series_list = self._select_series(conditions, study, all_matched)
                if series_list and len(series_list) > 0:
                    for series in series_list:
                        selected_series = SelectedSeries(
                            selection_name, series, None
                        )  # No Image obj yet.
                        study_selected_series.add_selected_series(selected_series)

            if len(study_selected_series.selected_series) > 0:
                study_selected_series_list.append(study_selected_series)

        return study_selected_series_list

    def _load_rules(self):
        return json_loads(self._rules_json_str) if self._rules_json_str else None

    def _select_all_series(self, dicom_study_list: List[DICOMStudy]) -> List[StudySelectedSeries]:
        """Select all series in studies

        Returns:
            list: list of StudySelectedSeries objects
        """

        study_selected_series_list = []
        for study in dicom_study_list:
            logging.info(f"Working on study, instance UID: {study.StudyInstanceUID}")
            study_selected_series = StudySelectedSeries(study)
            for series in study.get_all_series():
                logging.info(f"Working on series, instance UID: {str(series.SeriesInstanceUID)}")
                selected_series = SelectedSeries(
                    "", series, None
                )  # No selection name or Image obj.
                study_selected_series.add_selected_series(selected_series)
            study_selected_series_list.append(study_selected_series)
        return study_selected_series_list

    def _select_series(
        self, attributes: dict, study: DICOMStudy, all_matched=False
    ) -> List[DICOMSeries]:
        """Finds series whose attributes match the given attributes.

        Args:
            attributes (dict): Dictionary of attributes for matching
            all_matched (bool): Gets all matched series in a study. Defaults to False for first match only.

        Returns:
            List of DICOMSeries. At most one element if all_matched is False.
        """
        assert isinstance(attributes, dict), '"attributes" must be a dict.'

        logging.info(
            f"Searching study, : {study.StudyInstanceUID}\n  # of series: {len(study.get_all_series())}"
        )
        study_attr = self._get_instance_properties(study)

        found_series = []
        for series in study.get_all_series():
            logging.info(f"Working on series, instance UID: {series.SeriesInstanceUID}")

            # Combine Study and current Series properties for matching
            series_attr = self._get_instance_properties(series)
            series_attr.update(study_attr)

            matched = True
            # Simple matching on attribute value
            for key, value_to_match in attributes.items():
                logging.info(f"On attribute: {key!r} to match value: {value_to_match!r}")
                # Ignore None
                if not value_to_match:
                    continue
                # Try getting the attribute value from Study and current Series prop dict
                attr_value = series_attr.get(key, None)
                logging.info(f"    Series attribute {key} value: {attr_value}")

                # If not found, try the best at the native instance level for string VR
                # This is mainly for attributes like ImageType
                if not attr_value:
                    try:
                        # Can use some enhancements, especially multi-value where VM > 1
                        elem = series.get_sop_instances()[0].get_native_sop_instance()[key]
                        if elem.VM > 1:
                            attr_value = [
                                elem.repval
                            ]  # repval: str representation of the element’s value
                        else:
                            attr_value = elem.value  # element's value

                        series_attr.update({key: attr_value})
                    except Exception:
                        logging.info(f"        Attribute {key} not at instance level either.")

                if not attr_value:
                    matched = False
                elif isinstance(attr_value, numbers.Number):
                    matched = value_to_match == attr_value
                elif isinstance(attr_value, str):
                    matched = attr_value.casefold() == (value_to_match.casefold())
                    if not matched:
                        # For str, also try RegEx search to check for a match anywhere in the string
                        # unless the user constrains it in the expression.
                        logging.info("Series attribute string value did not match. Try regEx.")
                        if re.search(value_to_match, attr_value, re.IGNORECASE):
                            matched = True
                elif isinstance(attr_value, list):
                    # Assume multi value string attributes
                    meta_data_list = str(attr_value).lower()
                    if isinstance(value_to_match, list):
                        value_set = {str(element).lower() for element in value_to_match}
                        matched = all(val in meta_data_list for val in value_set)
                    elif isinstance(value_to_match, (str, numbers.Number)):
                        matched = str(value_to_match).lower() in meta_data_list
                else:
                    raise NotImplementedError(
                        f"Not support for matching on this type: {type(value_to_match)}"
                    )

                if not matched:
                    logging.info("This series does not match the selection conditions.")
                    break

            if matched:
                logging.info(f"Selected Series, UID: {series.SeriesInstanceUID}")
                found_series.append(series)

                if not all_matched:
                    return found_series

        return found_series

    @staticmethod
    def _get_instance_properties(obj: object):
        if not obj:
            return {}
        else:
            return {
                x: getattr(obj, x, None)
                for x in type(obj).__dict__
                if isinstance(type(obj).__dict__[x], property)
            }


# Module functions
# Helper function to get console output of the selection content when testing the script
def _print_instance_properties(obj: object, pre_fix: str = "", print_val=True):
    print(f"{pre_fix}Instance of {type(obj)}")
    for attribute in [x for x in type(obj).__dict__ if isinstance(type(obj).__dict__[x], property)]:
        attr_val = getattr(obj, attribute, None)
        print(f"{pre_fix}  {attribute}: {type(attr_val)} {attr_val if print_val else ''}")


def test():
    from pathlib import Path

    from operators.medical_imaging.dicom_data_loader_operator import DICOMDataLoaderOperator

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("../../../inputs/spleen_ct/dcm").absolute()

    fragment = Fragment()
    loader = DICOMDataLoaderOperator(fragment, name="loader_op")
    selector = DICOMSeriesSelectorOperator(fragment, name="selector_op")
    study_list = loader.load_data_to_studies(data_path)
    sample_selection_rule = json_loads(Sample_Rules_Text)
    print(f"Selection rules in JSON:\n{sample_selection_rule}")
    study_selected_seriee_list = selector.filter(sample_selection_rule, study_list)

    for sss_obj in study_selected_seriee_list:
        _print_instance_properties(sss_obj, pre_fix="", print_val=False)
        study = sss_obj.study
        pre_fix = "  "
        print(f"{pre_fix}==== Details of the study ====")
        _print_instance_properties(study, pre_fix, print_val=False)
        print(f"{pre_fix}==============================")

        # The following commented code block accesses and prints the flat list of all selected series.
        # for ss_obj in sss_obj.selected_series:
        #     pre_fix = "    "
        #     _print_instance_properties(ss_obj, pre_fix, print_val=False)
        #     pre_fix = "      "
        #     print(f"{pre_fix}==== Details of the series ====")
        #     _print_instance_properties(ss_obj, pre_fix)
        #     print(f"{pre_fix}===============================")

        # The following block uses hierarchical grouping by selection name, and prints the list of series for each.
        for selection_name, ss_list in sss_obj.series_by_selection_name.items():
            pre_fix = "    "
            print(f"{pre_fix}Selection name: {selection_name}")
            for ss_obj in ss_list:
                pre_fix = "        "
                _print_instance_properties(ss_obj, pre_fix, print_val=False)
                print(f"{pre_fix}==== Details of the series ====")
                _print_instance_properties(ss_obj, pre_fix)
                print(f"{pre_fix}===============================")

        print(
            f"  A total of {len(sss_obj.selected_series)} series selected for study {study.StudyInstanceUID}"
        )


# Sample rule used for testing
Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "CT Series 1",
            "conditions": {
                "StudyDescription": "(?i)^Spleen",
                "Modality": "(?i)CT",
                "SeriesDescription": "(?i)^No series description|(.*?)"
            }
        },
        {
            "name": "CT Series 2",
            "conditions": {
                "Modality": "CT",
                "BodyPartExamined": "Abdomen",
                "SeriesDescription" : "Not to be matched"
            }
        }
    ]
}
"""

if __name__ == "__main__":
    test()
