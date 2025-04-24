#
# Place the license header here
#

import holoscan as hs


class App(hs.core.Application):
    def compose(self):
        # Add your operators here
        print("Hello, Holoscan!")


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
