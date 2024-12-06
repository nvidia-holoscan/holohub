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

import time
from datetime import datetime, timedelta

import requests


class TokenProvider:
    """This class manages authorization token with OAuth2 server-to-server workflow.

    This class is initialized with a client id and secret along with a OAuth2
    authorization service URL, acquires the authorization token using the two legged
    server-to-server workflow, and re-acquires the token if expired. Refresh token
    is not needed or available in the OAuth2 two legged workflow.

    The formatted authorization token with token type is exposed as a property.
    """

    GRANT_TYPE = "client_credentials"
    TOKEN_TYPE = "Bearer"
    SCOPE = "openid"
    EXPIRES_IN = 3600

    def __init__(
        self,
        oauth_url: str,
        client_id: str,
        client_secret: str,
        verify_cert: bool = True,
    ):

        self.oauth_url = oauth_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.verify_cert = verify_cert  # True for verifying server cert
        self.expires_in_seconds_default = TokenProvider.EXPIRES_IN

        self._expires_datetime = datetime.now()
        self._token = None
        self._token_type = TokenProvider.TOKEN_TYPE
        self._grant_type = TokenProvider.GRANT_TYPE
        self._scope = TokenProvider.SCOPE

    @property
    def token(self):
        """Property representing the authorization token only

        Returns:
            str: token string
        """
        # If expired, refresh or get new token, otherwise, return existing
        if (not self._token) or (datetime.now() > self._expires_datetime):
            # Acquire access token again if none or expired.
            # The two legged workflow does not have refresh token.
            self.get_token()
        return self._token

    @property
    def authorization_header(self):
        """Property representing the Authorization header content, <type> <token>

        Return:
            str: content of Authorization header
        """
        token_only = self.token  # makes sure token is acquired and not expired
        return f"{self._token_type} {token_only}"

    def get_token(self):
        request_time = datetime.now()
        post_data = {
            "grant_type": self._grant_type,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self._scope,
        }

        max_retry = 3
        for i in range(max_retry):
            try:
                response = requests.post(self.oauth_url, data=post_data, verify=self.verify_cert)
                response.raise_for_status()

                token_json = response.json()
                # Parse the required attributes
                self._token = token_json["access_token"]
                self._token_type = token_json["token_type"]

                # Parse the recommended attributes
                expires_in = int(token_json.get("expires_in", self.expires_in_seconds_default))
                self._expires_datetime = request_time + timedelta(seconds=expires_in)

                self._refresh_token = token_json.get("refresh_token", None)
                self._scope = token_json.get("scope", self._scope)
                break
            except Exception as ex:
                if (i + 1) >= max_retry:
                    raise Exception(f"Failed to get the token due to exception:\n{ex}")
                time.sleep(3)
