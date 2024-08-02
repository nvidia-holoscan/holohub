if(NOT EXISTS "${PVA_UNSHARP_MASK_LIB_DEST}")
  # Download libpva_unsharp_mask.a using curl
  message(STATUS "libpva_unsharp_mask.a not found in source directory. Downloading from ${PVA_UNSHARP_MASK_URL} using curl")
  execute_process(COMMAND curl -L ${PVA_UNSHARP_MASK_URL} -o ${PVA_UNSHARP_MASK_LIB_DEST}
                  RESULT_VARIABLE result
                  OUTPUT_QUIET)
  if(NOT result EQUAL "0")
    message(FATAL_ERROR "Error downloading libpva_unsharp_mask.a using curl")
  endif()
  # Check if the downloaded file contains a "File not found" error message
  file(READ ${PVA_UNSHARP_MASK_LIB_DEST} contents)
  if(contents MATCHES "\"status\" : 404")
    message(FATAL_ERROR "Downloaded file contains a 'File not found' error. Please check the URL and try again.")
  endif()
  # Download cupva_allowlist_pva_unsharp_mask using curl
  message(STATUS "Downloading cupva_allowlist_pva_unsharp_mask from ${CUPVA_ALLOWLIST_URL} using curl")
  execute_process(COMMAND curl -L ${CUPVA_ALLOWLIST_URL} -o ${CUPVA_ALLOWLIST_DEST}
                  RESULT_VARIABLE result_allowlist
                  OUTPUT_QUIET)
  if(NOT result_allowlist EQUAL "0")
    message(FATAL_ERROR "Error downloading cupva_allowlist_pva_unsharp_mask using curl")
  endif()
  # Check if the downloaded file contains a "File not found" error message
  file(READ ${CUPVA_ALLOWLIST_DEST} contents_allowlist)
  if(contents_allowlist MATCHES "\"status\" : 404")
    message(FATAL_ERROR "Downloaded cupva_allowlist_pva_unsharp_mask contains a 'File not found' error. Please check the URL and try again.")
  endif()
endif()
