#
# Place the license header here
#

import holoscan as hs


class App(hs.Application):
    def compose(self):
        # Add your operators here
        pass


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
