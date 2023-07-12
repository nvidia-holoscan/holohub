import { ChevronRightIcon } from '@heroicons/react/20/solid'

const Hero = () => {
  return (
    <div className="mx-auto max-w-2xl py-24">
    <div className="hidden sm:mb-8 sm:flex sm:justify-center">
      <div className="relative rounded-full px-3 py-1  mt-2 text-sm leading-6 text-gray-600 ring-1 ring-gray-900/10 hover:ring-gray-900/20">
        What's New? &nbsp;
        <a href="https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v0.5.0" className="font-semibold text-lime-500"
        target="_blank">
          <span className="absolute inset-0" aria-hidden="true" />
          Just shipped NVIDIA Holoscan SDK v0.5.0 <span aria-hidden="true">&rarr;</span>
        </a>
      </div>
    </div>
    <div className="text-center">
      <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
        NVIDIA HoloHub
      </h1>
      <p className="mt-6 text-lg leading-8 text-gray-600">
      HoloHub is a central repository for users and developers of extensions and applications for the Holoscan Platform to share reusable components and sample applications.
      </p>
      <div className="mt-10 flex items-center justify-center gap-x-6">
        <a
          href="https://github.com/nvidia-holoscan/holohub"
          className="rounded-md bg-lime-500 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-lime-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-lime-600"
          target="_blank"
        >
          HoloHub GitHub
        </a>
      </div>
    </div>
  </div>
  )
}

export default Hero;
