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


import base64
import json
import logging
import pydantic
import re
from typing import Any, List, Optional, Dict, Tuple
from fhir.resources.R4B import construct_fhir_element
from fhir.resources.R4B import FHIRAbstractModel
from fhir.resources.R4B.allergyintolerance import AllergyIntolerance, AllergyIntoleranceReaction
from fhir.resources.R4B.observation import Observation, ObservationComponent
from fhir.resources.R4B.patient import Patient
from fhir.resources.R4B.humanname import HumanName
from fhir.resources.R4B.familymemberhistory import FamilyMemberHistory
from fhir.resources.R4B.encounter import Encounter
from fhir.resources.R4B.condition import Condition
from fhir.resources.R4B.coding import Coding
from fhir.resources.R4B.diagnosticreport import DiagnosticReport
from fhir.resources.R4B.attachment import Attachment
from fhir.resources.R4B.documentreference import DocumentReference, DocumentReferenceContent
from fhir.resources.R4B.medication import Medication
from fhir.resources.R4B.medicationrequest import MedicationRequest
from fhir.resources.R4B.medicationstatement import MedicationStatement
from fhir.resources.R4B.careplan import CarePlan
from fhir.resources.R4B.procedure import Procedure
from fhir.resources.R4B.immunization import Immunization
from fhir.resources.R4B.imagingstudy import ImagingStudy, ImagingStudySeries
from fhir.resources.R4B.servicerequest import ServiceRequest


logger = logging.getLogger("FHIRResourceSanitizer")

class FHIRResourceSanitizer:
    """
    A helper class that extracts/transforms a single FHIR document/record.
    """

    @staticmethod
    def sanitize(data: Dict) -> Optional[Dict]:
        """The sanitize method extracts wanted data, removes noises and transforms the data structure
            so the AI model can better understand the details of a given FHIR record.

        Args:
            data (Dict): a single FHIR document in python dictionary format.
            The dictionary shall have the following schema:
            {
                "fullUrl": "urn:uuid:38d90533-e6d2-3254-1d16-0744d317a3b2",
                "resource": {
                    # contains the actual FHIR resource document
                },
                "request": {
                    "method": "POST",
                    "url": "[FHIR Resource Type]"
                }
            }

        Raises:
            NotImplementedError: if the FHIR resource type is not supported.
                Note: unsupported resource types shall be added to the map.

        Returns:
            Optional[Dict]: a sanitized/transformed FHIR document/record.
        """
        resource = data.get("resource", None)
        if resource is None:
            return None

        resource_type = resource.get("resourceType", None)
        if resource_type is None:
            return None

        map = {
            "AllergyIntolerance": FHIRResourceSanitizer.allergy_intolerance,
            "CarePlan": FHIRResourceSanitizer.care_plan,
            "Condition": FHIRResourceSanitizer.condition,
            "DiagnosticReport": FHIRResourceSanitizer.diagnostic_report,
            "DocumentReference": FHIRResourceSanitizer.document_reference,
            "Encounter": FHIRResourceSanitizer.encounter,
            "FamilyMemberHistory": FHIRResourceSanitizer.family_member_history,
            "Medication": FHIRResourceSanitizer.medication,
            "MedicationRequest": FHIRResourceSanitizer.medication_request,
            "MedicationStatement": FHIRResourceSanitizer.medication_statement,
            "ImagingStudy": FHIRResourceSanitizer.imaging_study,
            "Immunization": FHIRResourceSanitizer.immunization,
            "Observation": FHIRResourceSanitizer.observation,
            "Patient": FHIRResourceSanitizer.patient,
            "Procedure": FHIRResourceSanitizer.procedure,
            "ServiceRequest": FHIRResourceSanitizer.service_request,
            # unsupported resource types
            "CareTeam": FHIRResourceSanitizer._no_op,
            "Claim": FHIRResourceSanitizer._no_op,
            "Device": FHIRResourceSanitizer._no_op,
            "ExplanationOfBenefit": FHIRResourceSanitizer._no_op,
            "Location": FHIRResourceSanitizer._no_op,
            "MedicationAdministration": FHIRResourceSanitizer._no_op,
            "Organization": FHIRResourceSanitizer._no_op,
            "Practitioner": FHIRResourceSanitizer._no_op,
            "PractitionerRole": FHIRResourceSanitizer._no_op,
            "Provenance": FHIRResourceSanitizer._no_op,
            "SupplyDelivery": FHIRResourceSanitizer._no_op,
        }

        try:
            model = construct_fhir_element(resource_type, resource)
        # If using fhir.resources 0.7.1, then
        # pydantic.v1.error_wrappers.ValidationError
        except pydantic.error_wrappers.ValidationError:
            logger.error("Validation error while processing: ", resource)
            return None

        if resource_type in map:
            return map[resource_type](data, model)
        else:
            raise NotImplementedError(f"{resource_type} type, id={model.id}")

    @staticmethod
    def _no_op(original_data: Dict, model: Any) -> None:
        """For unsupported FHIR resources

        Returns:
            None
        """
        return None

    @staticmethod
    def imaging_study(original_data: Dict, model: ImagingStudy) -> Dict:
        """Handles FHIR resource type: ImagingStudy"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        resource["studyInstanceUids"] = [
            item.value.replace("urn:oid:", "")
            for item in model.identifier
            if item.system == "urn:dicom:uid"
        ]
        resource["numberOfSeries"] = model.numberOfInstances
        resource["numberOfInstances"] = model.numberOfSeries
        resource["series"] = FHIRResourceSanitizer._series(resource, model.series)

        return sanitized_data

    @staticmethod
    def _series(resource: Dict, series: ImagingStudySeries) -> Dict:
        output = []
        for s in series:
            new_series = {}
            new_series["uid"] = s.uid
            new_series["number"] = s.number
            new_series["modality"] = s.modality.code
            new_series["description"] = s.description
            new_series["numberOfInstances"] = s.numberOfInstances
            new_series["bodySite"] = s.bodySite.display
            new_series["instance"] = [
                {
                    "uid": item.uid,
                    "sopClass": item.sopClass.code.replace("urn:oid:", ""),
                    "number": item.number,
                }
                for item in s.instance
            ]
            output.append(new_series)
        return output

    @staticmethod
    def allergy_intolerance(original_data: Dict, model: AllergyIntolerance) -> Dict:
        """Handles FHIR resource type: AllergyIntolerance"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        resource["clinical_status"] = ",".join(
            [FHIRResourceSanitizer._get_coding_value(item) for item in model.clinicalStatus.coding]
        )
        resource["verification_status"] = ",".join(
            [
                FHIRResourceSanitizer._get_coding_value(item)
                for item in model.verificationStatus.coding
            ]
        )
        resource["type"] = model.type
        resource["criticality"] = model.criticality
        resource["allergy_intolerance"] = [
            {"code": item.code, "display": item.display} for item in model.code.coding
        ]
        if model.reaction:
            resource["reaction"] = FHIRResourceSanitizer._allergy_reaction(model.reaction)
        return sanitized_data

    @staticmethod
    def immunization(original_data: Dict, model: Immunization) -> Dict:
        """Handles FHIR resource type: Immunization"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        resource["vaccines"] = model.vaccineCode.text
        resource["date"] = model.occurrenceDateTime.isoformat()
        resource["location"] = model.location.display
        return sanitized_data

    @staticmethod
    def procedure(original_data: Dict, model: Procedure) -> Dict:
        """Handles FHIR resource type: Procedure"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        resource["procedure"] = [item.display for item in model.code.coding]
        resource["date"] = model.performedPeriod.start.isoformat() if model.performedPeriod else None
        if model.location and model.location.display:
            resource["location"] = model.location.display
        if model.reasonReference:
            resource["reasons"] = [item.display for item in model.reasonReference]
        if model.bodySite:
            resource["body_site"] = [
                FHIRResourceSanitizer._get_coding_value(item)
                for category in model.bodySite
                for item in category.coding
            ]
        if model.outcome and model.outcome.text:
            resource["outcome"] = model.outcome.text
        if model.report:
            resource["report"] = {item.display: item.reference for item in model.report}

        return sanitized_data

    @staticmethod
    def service_request(original_data: Dict, model: ServiceRequest) -> Dict:
        """Handles FHIR resource type: Procedure"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)

        resource["status"] = model.status
        resource["intent"] = model.intent
        resource["request_order"] = [item.display for item in model.code.coding]
        resource["occurrence"] = model.occurrenceDateTime.isoformat() if model.occurrenceDateTime else "N/A"
        resource["performer"] = [item.reference for item in model.performer] if model.performer else "N/A"
        resource["reason"] = [item.display for code in model.reasonCode for item in code.coding]

        return sanitized_data

    @staticmethod
    def care_plan(original_data: Dict, model: CarePlan) -> Dict:
        """Handles FHIR resource type: CarePlan"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        resource["date"] = model.period.start.isoformat()
        resource["text"] = (
            FHIRResourceSanitizer._remove_html_tags(model.text.div) if model.text.div else None
        )
        resource["categories"] = [
            FHIRResourceSanitizer._get_coding_value(item)
            for category in model.category
            for item in category.coding
        ]
        if model.activity:
            resource["activities"] = [item.detail.code.text for item in model.activity]
        return sanitized_data

    @staticmethod
    def medication(original_data: Dict, model: Medication) -> Dict:
        """Handles FHIR resource type: Medication"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        resource["medication"] = [
            FHIRResourceSanitizer._get_coding_value(item) for item in model.code.coding
        ]
        return sanitized_data

    @staticmethod
    def medication_request(original_data: Dict, model: MedicationRequest) -> Dict:
        """Handles FHIR resource type: MedicationRequest"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        resource["intent"] = model.intent

        if model.medicationCodeableConcept:
            resource["medication"] = [
                FHIRResourceSanitizer._get_coding_value(item)
                for item in model.medicationCodeableConcept.coding
            ]
        if model.medicationReference:
            resource["medication_reference"] = model.medicationReference.reference
        resource["requester"] = model.requester.display
        resource["dosage"] = "not supported"
        return sanitized_data

    @staticmethod
    def medication_statement(original_data: Dict, model: MedicationStatement) -> Dict:
        """Handles FHIR resource type: MedicationStatement"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)

        if model.medicationCodeableConcept:
            resource["medication"] = [
                FHIRResourceSanitizer._get_coding_value(item)
                for item in model.medicationCodeableConcept.coding
            ]
        resource["dosage"] = [item.text for item in model.dosage]
        resource["effective_date_time"] = model.effectiveDateTime.isoformat() if model.effectiveDateTime else "N/A"
        return sanitized_data

    @staticmethod
    def document_reference(original_data: Dict, model: DocumentReference) -> Dict:
        """Handles FHIR resource type: DocumentReference"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)

        resource["type"] = [
            FHIRResourceSanitizer._get_coding_value(item) for item in model.type.coding
        ]
        resource["categories"] = [
            FHIRResourceSanitizer._get_coding_value(item)
            for category in model.category
            for item in category.coding
        ]
        resource["authors"] = [item.display for item in model.author]
        resource["document_reference"] = FHIRResourceSanitizer._parse_document_references(
            model.content
        )

        resource["date"] = model.context.period.start.isoformat()

        return sanitized_data

    @staticmethod
    def diagnostic_report(original_data: Dict, model: DiagnosticReport) -> Dict:
        """Handles FHIR resource type: DiagnosticReport"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)

        if model.conclusion:
            resource["conclusion"] = model.conclusion

        resource["date"] = model.effectiveDateTime.isoformat()
        if model.performer:
            resource["performers"] = [item.display for item in model.performer]
        if model.category:
            resource["categories"] = [
                FHIRResourceSanitizer._get_coding_value(item)
                for category in model.category
                for item in category.coding
            ]
        resource["codes"] = [
            FHIRResourceSanitizer._get_coding_value(code) for code in model.code.coding
        ]
        if model.presentedForm:
            resource["reports"] = []
            attachments = FHIRResourceSanitizer._parse_attachments(model.presentedForm)
            for attachment in attachments:
                resource["reports"].append(attachment)

        if model.media:
            resource["key_images"] = []
            for report_media in model.media:
                resource["key_images"].append((report_media.comment, report_media.link.reference))

        if model.result:
            resource["result_references"] = {item.display: item.reference for item in model.result}

        return sanitized_data

    @staticmethod
    def condition(original_data: Dict, model: Condition) -> Dict:
        """Handles FHIR resource type: Condition"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        resource["clinical_status"] = ",".join(
            [FHIRResourceSanitizer._get_coding_value(item) for item in model.clinicalStatus.coding]
        )
        resource["verification_status"] = ",".join(
            [
                FHIRResourceSanitizer._get_coding_value(item)
                for item in model.verificationStatus.coding
            ]
        )

        if model.category:
            resource["categories"] = ",".join(
                [
                    FHIRResourceSanitizer._get_coding_value(item)
                    for category in model.category
                    for item in category.coding
                ]
            )

        resource["condition"] = (
            model.code.text
            if model.code.text
            else ",".join(item.display for item in model.code.coding)
        )

        if model.bodySite:
            resource["body_site"] = ",".join(
                code.display for item in model.bodySite for code in item.coding
            )
        resource["onset_dateTime"] = (
            model.onsetDateTime.isoformat() if model.onsetDateTime else None
        )
        resource["abatement_dateTime"] = (
            model.abatementDateTime.isoformat() if model.abatementDateTime else None
        )
        resource["recorded_date"] = model.recordedDate.isoformat() if model.recordedDate else None
        if model.note:
            resource["notes"] = [item.text for item in model.note]

        if model.stage:
            resource["stage"] = [
                code.display for item in model.stage for code in item.summary.coding
            ]
        return sanitized_data

    @staticmethod
    def encounter(original_data: Dict, model: Encounter) -> Dict:
        """Handles FHIR resource type: Encounter"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        
        if model.type:
            resource["type"] = [
                FHIRResourceSanitizer._get_coding_value(code)
                for item in model.type
                for code in item.coding
            ]
        resource["class"] = model.class_fhir.code

        participants = {}
        if model.participant:
            for participant in model.participant:
                if participant.type:
                    for type in participant.type:
                        participants[type.text] = participant.individual.display

        resource["participants"] = participants
        resource["date"] = model.period.start.isoformat()
        return sanitized_data

    @staticmethod
    def family_member_history(original_data: Dict, model: FamilyMemberHistory) -> Dict:
        """Handles FHIR resource type: FamilyMemberHistory"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        relationships = []

        for relationship in model.relationship.coding:
            relationships.append(relationship.display)
        resource["relationship"] = relationships
        resource["deceased"] = model.deceasedBoolean

        conditions = []

        for condition in model.condition:
            for condition_code in condition.code.coding:
                conditions.append(condition_code.display)
        resource["conditions"] = conditions

        return sanitized_data

    @staticmethod
    def patient(original_data: Dict, model: Patient) -> Dict:
        """Handles FHIR resource type: Patient"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        resource["names"] = []
        for name in model.name:
            resource["names"].append(FHIRResourceSanitizer._get_name(name))
        resource["birth_date"] = model.birthDate.isoformat()
        resource["gender"] = model.gender
        resource["marital_status"] = FHIRResourceSanitizer._get_marital_status(model.maritalStatus)
        return sanitized_data

    @staticmethod
    def observation(original_data: Dict, model: Observation) -> Dict:
        """Handles FHIR resource type: Observation"""
        sanitized_data, resource = FHIRResourceSanitizer._new_resource(original_data, model)
        resource["status"] = model.status

        if model.category:
            resource["categories"] = ",".join(
                [
                    FHIRResourceSanitizer._get_coding_value(item)
                    for category in model.category
                    for item in category.coding
                ]
            )
        resource["observation"] = FHIRResourceSanitizer._get_observation_values(model)
        if model.effectiveDateTime:
            resource["date"] = model.effectiveDateTime.isoformat()

        return sanitized_data

    @staticmethod
    def _get_observation_values(model: FHIRAbstractModel):
        """Extracts the observation type and value from code and value sections."""
        if not isinstance(model, ObservationComponent) and model.component:
            values = []
            for item in model.component:
                values.append(FHIRResourceSanitizer._get_observation_values(item))
            return values
        else:
            if model.code.text:
                text = model.code.text
            else:
                text = ",".join(item.display for item in model.code.coding)

            if model.valueQuantity:
                return f"{text}: {model.valueQuantity.value} {model.valueQuantity.unit.replace('[', '').replace(']', '')}"
            elif model.valueCodeableConcept:
                if len(model.valueCodeableConcept.coding) == len(model.code.coding):
                    values = []
                    for index in range(len(model.valueCodeableConcept.coding)):
                        values.append(
                            f"{model.code.coding[index].display}: {model.valueCodeableConcept.coding[index].display}"
                        )
                    return values
                else:
                    return f"{text}: {model.valueCodeableConcept.text}"
            elif model.valueString:
                return f"{text}: {model.valueString}"
            elif model.valueBoolean:
                return f"{text}: {model.valueBoolean}"
            else:
                raise NotImplementedError("value type")

    @staticmethod
    def _parse_document_references(document_references: List[DocumentReferenceContent]) -> List:
        """Attempts to parse given document references.

        Raises:
            NotImplementedError: if the format of the document reference is unsupported.

        Returns:
            List: list of documents parsed
        """
        docs = {}
        for doc_ref in document_references:
            format = doc_ref.format.code
            if format == "urn:ihe:iti:xds:2017:mimeTypeSufficient":
                type, data = FHIRResourceSanitizer._parse_attachment(doc_ref.attachment)
                docs[type] = data
            else:
                raise NotImplementedError(f"Unknown format {format}")

        return docs

    @staticmethod
    def _parse_attachments(attachments: List[Attachment]) -> Dict[str, Any]:
        """Attempts to parse attachments.

        Returns:
            Dict[str, Any]: a dictionary where the key is MIME type of the document and value is the actual data.
            See _parse_attachment.
        """
        docs = []
        for attachment in attachments:
            type, data = FHIRResourceSanitizer._parse_attachment(attachment)
            docs.append({"type": type, "data": data})
        return docs

    @staticmethod
    def _parse_attachment(attachment: Attachment) -> Tuple[str, Any]:
        """Parses a single attachment.

        Returns:
            Tuple[str, Any]: A tuple where the first value is the content type and the
                             second value is the actual data.
                             When content type is text, the data is the actual decoded value.
                             Otherwise, the value is in the original form which could be base64 encoded string.
        """
        if attachment.contentType == "text/plain; charset=utf-8":
            return (attachment.contentType, base64.b64decode(attachment.data).decode("utf-8"))
        else:
            return (attachment.contentType, attachment.data)
            # raise NotImplementedError(f"Content type not supported: {attachment.contentType}")

    @staticmethod
    def _get_coding_value(coding: Coding) -> str:
        """Returns the display value of a given Coding object when available. Otherwise, return the actual code."""
        return coding.display if coding.display else coding.code

    @staticmethod
    def _get_name(name: HumanName) -> str:
        """Returns a formatted name of a given HumanName"""

        def _prefix(prefixes) -> str:
            return ",".join(prefixes) if prefixes else ""

        return f"{_prefix(name.prefix)} {name.family}, {' '.join(name.given)}".strip()

    @staticmethod
    def _get_marital_status(marital_status: str) -> str:
        """Converts abbreviated martial status to actual meaning."""
        if marital_status and marital_status.text:
            mapping = {
                "A": "Annulled",
                "D": "Divorced",
                "I": "Interlocutory",
                "L": "Legally Separated",
                "M": "Married",
                "C": "Common Law",
                "P": "Polygamous",
                "T": "Domestic partner",
                "U": "Unmarried",
                "S": "Never Married",
                "W": "Widowed",
            }

            text = marital_status.text.upper()
            return mapping.get(text, text)
        return "Unknown"

    @staticmethod
    def _new_resource(original_data: Dict, model: FHIRAbstractModel) -> Dict:
        """Creates a sanitized resource dictionary with common/shared attributes"""
        sanitized_data = {}
        resource = {}
        sanitized_data["resource"] = resource
        sanitized_data["fullUrl"] = original_data.get("fullUrl", "N/A")
        resource["resourceType"] = model.resource_type
        resource["id"] = model.id
        return sanitized_data, resource

    @staticmethod
    def _allergy_reaction(reactions: AllergyIntoleranceReaction) -> Dict:
        """Parses AllergyIntoleranceReaction"""
        data = {}
        for reaction in reactions:
            texts = [item.text for item in reaction.manifestation]
            data[",".join(texts)] = reaction.severity
        return data

    @staticmethod
    def _remove_html_tags(text) -> str:
        """Removes HTML tags"""
        tags = re.compile("<.*?>")
        return re.sub(tags, "", text)


if __name__ == "__main__":
    import glob

    path = "/home/vicchang/sc/fhir-data/John Doe/servicerequest.json"

    for file in glob.glob(path):
        print("==========================================================")
        print(f"Processing file {file}..........................")
        with open(file) as f:
            sample = json.load(f)

        entries = sample.get("entry")

        if entries:
            for entry in entries:
                # print(f'processing {entry}')
                x = FHIRResourceSanitizer.sanitize(entry)
                # print(json.dumps(x))
                print(x)
