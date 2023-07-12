import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const Card = ({ data, openModal }) => {
  const { metadata, readme, source_folder } = data;
  const { name, language, authors, tags } = metadata;
  const formattedSourceFolder = source_folder.charAt(0).toUpperCase() + source_folder.slice(1, -1);

  let sourceFolderColor = "bg-lime-50 text-lime-700";  
  if (formattedSourceFolder.toLowerCase() === "application") {
    sourceFolderColor = "bg-sky-50 text-sky-700";
  } else if (formattedSourceFolder.toLowerCase() === "operator") {
    sourceFolderColor = "bg-red-50 text-red-700";
  }

  return (
    <div className="flex max-w-xl flex-col items-start justify-between border-2 p-4 rounded-lg">
      <span className={`relative z-10 -top-8 -mb-5 rounded-md bg-neutral-50 my-1 px-2 py-1 text-xs font-medium text-neutral-700 ring-1 ring-inset ring-neutral-700/10 ${sourceFolderColor}`}>
        {formattedSourceFolder}
      </span>
      <div className="flex flex-wrap items-center gap-x-4 text-xs">
        {tags.slice(1).map((tag, index) => (
          <span
            key={index}
            className="relative z-10 rounded-md bg-lime-50 my-1 px-2 py-1 text-xs font-medium text-lime-700 ring-1 ring-inset ring-lime-700/10"
          >
            {tag}
          </span>
        ))}
      </div>
      <div className="group relative max-w-full grow">
        <h3 className="mt-3 text-lg font-semibold leading-6 text-gray-900 group-hover:text-gray-600">
          <span className="absolute inset-0" />
          {name}
        </h3>
        {language && <span>Language: {language}</span>}
        {authors && (
          <div className="relative flex items-center gap-x-4">
            <div className="leading-6">
              <p className="text-gray-900">
                Authors: {authors.map((author) => author.name).join(', ')}
              </p>
            </div>
          </div>
        )}
        <span>README:</span>
        <ReactMarkdown
          className="my-2 line-clamp-3 text-sm leading-6 text-gray-600 break-words"
          remarkPlugins={[remarkGfm]}
        >
          {readme}
        </ReactMarkdown>
      </div>
      <button
        onClick={() => openModal(data)}
        className="bg-lime-500 text-white px-4 py-2 mt-2 rounded hover:bg-lime-600"
      >
        View Details
      </button>
    </div>
  );
};

export default Card;
