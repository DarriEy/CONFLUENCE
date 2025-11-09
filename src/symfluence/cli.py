def main():
    # Delegate to our existing CLI
    try:
        from utils.cli.cli_manager import main as real_main
    except Exception as e:
        # Helpful error if the backend isn't packaged for some reason
        import sys, traceback
        sys.stderr.write(
            "symfluence: could not import utils.cli.cli_manager: "
            f"{e}\n"
            "Make sure 'utils' is a package and included in the wheel.\n"
        )
        traceback.print_exc()
        return 1
    return real_main()
