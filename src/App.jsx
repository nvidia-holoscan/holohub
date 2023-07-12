import React, { useEffect, useReducer, useState } from "react";
import { Transition, Dialog } from '@headlessui/react';
import { XMarkIcon } from "@heroicons/react/24/outline";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm'

import { reducer, initialState, actionTypes } from "./reducers";
import Card from "./Card";
import Hero from "./Hero";

const App = () => {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [searchTerm, setSearchTerm] = useState("");
  const [showClearButton, setShowClearButton] = useState(false);
  const [enabledTags, setEnabledTags] = useState([]);
  const [allTags, setAllTags] = useState([]);
  const [selectedData, setSelectedData] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Fetch data from the generated workflow JSON
  const fetchData = async () => {
    dispatch({ type: actionTypes.FETCH_DATA_START });

    try {
      const response = await fetch("aggregate_metadata.json");
      if (!response.ok) {
        throw new Error("Failed to fetch data");
      }

      const data = await response.json();
      dispatch({ type: actionTypes.FETCH_DATA_SUCCESS, payload: data });
      extractAllTags(data);
    } catch (error) {
      dispatch({ type: actionTypes.FETCH_DATA_ERROR, payload: error.message });
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Apply filters when search term, enabled tags, or data changes
  useEffect(() => {
    applyFilter();
  }, [searchTerm, enabledTags, state.data]);

  // Handle initial hash change and subsequent hash changes
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash;
      if (hash && hash.startsWith("#")) {
        const cardId = hash.substring(1);
        const cardIdParts = cardId.split("_");
        const cardIdPart0 = cardIdParts[0];
        let applicationName = "";
        debugger;
        if (cardIdPart0 === "model") {
          applicationName = cardIdParts.slice(1).join("_");
        } else {
          applicationName = cardIdParts.slice(2).join("_");
        }
        const selectedCard = state.data.find((card) => {
          if (card.metadata.language) {
            return (card.metadata.language.toLowerCase() === cardIdPart0 &&
            card.application_name === applicationName)
          } else {
            return card.application_name === applicationName;
          }
        });

        if (selectedCard) {
          debugger;
          openModal(selectedCard);
        }
      } else {
        closeModal();
      }
    };

    const handleInitialHashChange = () => {
      setTimeout(() => {
        handleHashChange();
      }, 100);
    };

    handleInitialHashChange(); // Call the function when the component mounts to handle initial hash

    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, [state.data]);

  // Handle search input change
  const handleSearch = (event) => {
    const value = event.target.value;
    setSearchTerm(value);
    setShowClearButton(value.length > 0);
  };

  const clearSearchTerm = () => {
    setSearchTerm("");
    setShowClearButton(false);
  };  

  // Handle tag click
  const handleTagClick = (tag) => {
    if (enabledTags.includes(tag)) {
      setEnabledTags(enabledTags.filter((t) => t !== tag));
    } else {
      setEnabledTags([...enabledTags, tag]);
    }
  };

  // Extract all tags from the data
  const extractAllTags = (data) => {
    const tagsCountMap = new Map();

    data.forEach((item) => {
      const { metadata, source_folder } = item;
      const { tags } = metadata;
  
      const formattedSourceFolder = source_folder
      .charAt(0)
      .toUpperCase() + source_folder.slice(1, -1);
  
      tags.unshift(formattedSourceFolder); // Include source_folder as a tag
  
      tags.forEach((tag) => {
        if (tagsCountMap.has(tag)) {
          tagsCountMap.set(tag, tagsCountMap.get(tag) + 1);
        } else {
          tagsCountMap.set(tag, 1);
        }
      });
    });

    const filteredTags = Array.from(tagsCountMap.entries()).filter(
      ([, count]) => count >= 4
    );
    const extractedTags = filteredTags.map(([tag]) => tag).sort();
    setAllTags(extractedTags);
  };

  // Apply filter based on search term and enabled tags
  const applyFilter = () => {
    const filteredData = state.data.filter((item) => {
      const { readme, metadata } = item;
      const { tags } = metadata;

      const lowerCaseSearchTerm = searchTerm.toLowerCase();
      const matchesSearchTerm =
        readme.toLowerCase().includes(lowerCaseSearchTerm) ||
        tags.some((tag) => tag.toLowerCase().includes(lowerCaseSearchTerm));


      if (searchTerm === "" && enabledTags.length === 0) {
        // No search term and no enabled tags, return all data
        return true;
      } else if (searchTerm === "" && enabledTags.length > 0) {
        // No search term but has enabled tags, match any enabled tag
        return enabledTags.some((tag) => tags.includes(tag));
      } else if (searchTerm !== "" && enabledTags.length === 0) {
        // Has search term but no enabled tags, match the search term
        return matchesSearchTerm;
      } else {
        // Has search term and enabled tags, match both search term and any enabled tag
        return (
          matchesSearchTerm && enabledTags.some((tag) => tags.includes(tag))
        );
      }
    });
    dispatch({ type: actionTypes.SET_FILTERED_DATA, payload: filteredData });
  };

  // Check if a tag is enabled
  const isTagEnabled = (tag) => enabledTags.includes(tag);

  // Get the filtered data
  const filteredData = state.filteredData || [];

  // Open modal and update URL hash
  const openModal = (data) => {
    setSelectedData(data);
    setIsModalOpen(true);
    let cardId;
    if (data.metadata.language) {
      const formattedLanguage = data.metadata.language.toLowerCase();
      cardId = `${formattedLanguage}_model_${data.application_name}`;
    } else {
      cardId = `model_${data.application_name}`;
    }
    window.location.hash = cardId;
  };

  // Close modal and remove hash from URL
  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedData(null);
    history.pushState("", document.title, window.location.pathname);
  };

  
  return (
    <>
      <Hero />
      <div className="container mx-auto px-4">
        <h1 className="text-2xl font-bold mb-4">HoloHub Applications</h1>

        <div className="relative my-2">
          <label
            htmlFor="name"
            className="absolute -top-2 left-2 inline-block bg-white px-1 text-xs font-medium text-gray-900"
          >
            Search
          </label>
          <input
            type="text"
            placeholder="Tool Segmentation..."
            value={searchTerm}
            onChange={handleSearch}
            className="block w-full rounded-md border-0 p-1.5 pl-2 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-lime-600 sm:text-sm sm:leading-6"
          />
          {showClearButton && (
            <button
              className="absolute top-1.5 right-2 flex items-center justify-center w-6 h-6 text-gray-500 hover:text-gray-600 focus:outline-none"
              onClick={clearSearchTerm}
            >
              <XMarkIcon className="h-6 w-6" aria-hidden="true" />
            </button>
          )}          
        </div>

        <div className="flex flex-wrap items-center gap-x-4 text-xs">
          <span>Popular Tags:</span>
          {allTags.map((tag) => (
            <span
              key={tag}
              className={`relative z-10 rounded-md my-1 px-2 py-1 text-xs font-medium cursor-pointer ${
                isTagEnabled(tag)
                  ? "bg-lime-500 text-white"
                  : "bg-gray-200 text-gray-800"
              }`}
              onClick={() => handleTagClick(tag)}
            >
              {tag}
            </span>
          ))}
        </div>

        {state.loading ? (
          <p className="my-4">Loading...</p>
        ) : state.error ? (
          <p className="my-4">Error: {state.error}</p>
        ) : filteredData.length > 0 ? (
          <div className="mx-auto grid max-w-2xl grid-cols-1 gap-x-8 gap-y-16 border-t border-gray-200 mb-8 pt-2 sm:mt-4 sm:pt-4 lg:mx-0 lg:max-w-none lg:grid-cols-3">
            {filteredData.map((jsonData, index) => (
              <Card key={index} data={jsonData} openModal={openModal} />
            ))}
          </div>
        ) : (
          <p className="my-4">No results found.</p>
        )}
      </div>
      {/* Modal */}
      <Transition.Root show={isModalOpen} as={React.Fragment}>
        <Dialog
          as="div"
          className="fixed inset-0 z-10 overflow-y-auto"
          onClose={closeModal}
          open={isModalOpen}
        >
          <div className="flex items-center justify-center min-h-screen">
            <Dialog.Overlay className="fixed inset-0 bg-black opacity-75" />

            <Transition.Child
              as={React.Fragment}
              enter="ease-out duration-200"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-150"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel
                className="bg-white rounded-lg shadow-lg border-2 border-gray-300 w-full max-h-[calc(80vh)] max-w-[calc(70vw)]"
                static
              >
               {selectedData && (
                  <>
                    <Dialog.Title className="text-lg font-bold p-4 border-b-2">
                      {selectedData.metadata.name}
                      <div className="absolute right-0 top-0 hidden pr-4 pt-4 sm:block">
                        <button
                          type="button"
                          className="rounded-md bg-white text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-grey-500 focus:ring-offset-2"
                          onClick={() => closeModal(true)}
                        >
                          <span className="sr-only">Close</span>
                          <XMarkIcon className="h-6 w-6" aria-hidden="true" />
                        </button>
                      </div>
                    </Dialog.Title>
                    <div className="w-full max-h-[calc(60vh)] overflow-y-auto p-4">
                      <Dialog.Description className="mb-4">
                      {/* Model Metadata */}
                      <section className="mb-6">
                      <h2 className="text-2xl font-semibold mb-2 text-lime-500">Model Metadata</h2>
                        <div className="space-y-2">
                          {selectedData.metadata.authors && (
                            <p>
                              <span className="font-semibold">Authors:</span>{" "}
                              {selectedData.metadata.authors
                                .map((author) => author.name)
                                .join(", ")}
                            </p>
                          )}
                          <p>
                            <span className="font-semibold">Version:</span>{" "}
                            {selectedData.metadata.version}
                          </p>     
                          {selectedData.metadata.holoscan_sdk?.minimum_required_version && (            
                            <p>
                              <span className="font-semibold">Minimum Required Holoscan Version:</span>{" "}
                              {selectedData.metadata.holoscan_sdk?.minimum_required_version}
                            </p>
                          )}
                          {selectedData.metadata.dependencies && (
                            <>
                              {selectedData.metadata.dependencies.libraries && (
                                <div>
                                  <p className="font-semibold">Library Dependencies:</p>
                                  <ul className="list-disc ml-6 space-y-1">
                                    {selectedData.metadata.dependencies.libraries.map((library) => (
                                      <li key={library.name}>{library.name}</li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              {selectedData.metadata.dependencies.gxf_extensions && (
                                <div>
                                  <p className="font-semibold">GXF Extension Dependencies:</p>
                                  <ul className="list-disc ml-6 space-y-1">
                                    {selectedData.metadata.dependencies.gxf_extensions.map((extension) => (
                                      <li key={extension.name}>{extension.name}</li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </>
                          )}
                        </div>
                      </section>
                      <div className="border-t border-gray-300 my-6" />
                        {/* README */}
                        <h2 className="text-2xl font-semibold mb-2 text-lime-500">Model README:</h2>
                        <ReactMarkdown
                          className="prose pre my-2 text-sm leading-6 text-gray-600 break-words"
                          remarkPlugins={[remarkGfm]}>
                            {selectedData.readme}
                        </ReactMarkdown>
                      </Dialog.Description>
                    </div>
                    <div className="flex justify-end p-4 border-t-2">
                      <a className="mx-2" href={`https://github.com/nvidia-holoscan/holohub/tree/main/${selectedData.source_folder}/${selectedData.application_name}`} target="_blank">
                        <button
                          className="bg-lime-500 text-white px-4 py-2 rounded hover:bg-lime-600 focus:outline-none focus:ring-2 focus:ring-lime-500"
                        >
                          Go to App on GitHub
                        </button>
                      </a>
                      <button
                        onClick={closeModal}
                        className="bg-lime-500 text-white px-4 py-2 rounded hover:bg-lime-600 focus:outline-none focus:ring-2 focus:ring-lime-500"
                      >
                        Close
                      </button>
                    </div>
                  </>
                )}
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </Dialog>
      </Transition.Root>
      {/* Footer */}
      <footer aria-labelledby="footer-heading">
        <h2 id="footer-heading" className="sr-only">Footer</h2>
        <div className="mx-auto max-w-7xl px-6 pb-8 lg:px-8">
          <div className="mt-16 border-t border-gray-900/10 lg:mt-24">
            <p className="mt-8 text-gray-500 text-center">NVIDIA HoloHub</p>
          </div>
        </div>
      </footer>
    </>
  );
};

export default App;
