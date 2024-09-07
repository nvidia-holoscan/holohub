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

import argparse
from datetime import timedelta
import logging
from urllib.parse import urlparse

from holoscan.core import Application
from holoscan.conditions import PeriodicCondition

from operators.ehr_query_llm.fhir.fhir_client_op import FhirClientOperator
from operators.ehr_query_llm.fhir.fhir_resource_sanitizer_op import FhirResourceSanitizerOp
from operators.ehr_query_llm.fhir.token_provider import TokenProvider
from operators.ehr_query_llm.zero_mq.subscriber_op import ZeroMQSubscriberOp
from operators.ehr_query_llm.zero_mq.publisher_op import ZeroMQPublisherOp

logging.getLogger("urllib3").setLevel(logging.WARNING)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parser for FHIR server endpoint and OAuth2 authorization credentials.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--fhir_url",
        type=str,
        help="The FHIR service URL.",
    )
    parser.add_argument(
        "--auth_url",
        type=str,
        default=None,
        help="The OAuth2 authorization service URL."
    )
    parser.add_argument(
        "--uid",
        type=str,
        help="The user/client ID used for requesting OAuth2 authorization token.",
        default=None,
    )
    parser.add_argument(
        "--secret",
        type=str,
        default=None,
        help="The user/client secret for requesting OAuth2 authorization token.",
    )
    parser.add_argument(
        "--verify_cert",
        action="store_true",  # SECURITY WARNING: verify server cert if flag provided.
        help="The user/client secret for requesting OAuth2 authorization token.",
    )
    args = parser.parse_args()
    return args

class FhirDemo(Application):
    def __init__(self, args):
        self.args = args
        super().__init__()

    def compose(self):
        # Basic validation of FHIR endpoint URL
        url_parsed = urlparse(self.args.fhir_url)
        if (not str(url_parsed.scheme).casefold().startswith('http')) or \
           (not url_parsed.netloc) or \
           (not url_parsed.hostname):
            raise ValueError("FHIR service URL is invalid.")

        token_provider = None
        if self.args.auth_url:
            # Create the auth token requester/provider
            token_provider = TokenProvider(
                oauth_url=self.args.auth_url,
                client_id=self.args.uid,
                client_secret=self.args.secret,
                verify_cert=self.args.verify_cert,
                )

        # Define the operators
        sub = ZeroMQSubscriberOp(
            self,
            PeriodicCondition(self, timedelta(milliseconds=1)),
            name="zmq",
            topic="ehr-request",
            queue_endpoint="tcp://localhost:5600",
            blocking=True,
        )
        fhir = FhirClientOperator(
            self,
            name="fhir",
            fhir_endpoint=self.args.fhir_url,
            token_provider=token_provider,
            verify_cert=self.args.verify_cert,
        )
        fhirsan = FhirResourceSanitizerOp(
            self,
            name="fhir-sanitizer",
        )
        pub = ZeroMQPublisherOp(
            self, name="pub", topic="ehr-response", queue_endpoint="tcp://*:5601"
        )

        # Define the workflow
        self.add_flow(sub, fhir)
        self.add_flow(fhir, fhirsan)
        self.add_flow(fhirsan, pub)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()
    app = FhirDemo(args=args)
    app.run()
