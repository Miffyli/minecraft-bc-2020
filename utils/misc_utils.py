#
# Miscellenous tools
#
from argparse import ArgumentParser
import inspect
from pprint import pprint


def parse_keyword_arguments(unparsed_args, class_object, debug=True):
    """
    Take unparsed arguments and a class object,
    check what keyword arguments class's __init__
    takes, create ArgumentParser object for them
    and parse them through. Return a dictionary
    mapping keywords to the parsed arguments, and remaining
    unparsed arguments.

    NOTE: class_object should have attribute "REQUIRED_ARGUMENTS",
        which is a list containing all keyword names that
        are required. If keyword is not required, the default
        value from __init__ definition will be used.

    Returns tuple of (
        dictionary mapping keyword names to parsed values,
        remaining unparsed args
    )

    If debug is true, print out what arguments are being gathered.
    """

    parser = ArgumentParser()
    arguments = inspect.signature(class_object.__init__)
    for argument in arguments.parameters.values():
        if argument.default is not argument.empty:
            parser.add_argument(
                "--{}".format(argument.name),
                type=type(argument.default),
                default=argument.default,
                required=argument.name in class_object.REQUIRED_ARGUMENTS
            )
    # Update unparsed args to detect duplicates of same variable name
    class_arguments, unparsed_args = parser.parse_known_args(unparsed_args)
    class_arguments = vars(class_arguments)

    if debug:
        print("\nArguments for {}".format(class_object.__name__))
        pprint(class_arguments, indent=4, width=1)
        # Coolio newline
        print("")

    return class_arguments, unparsed_args
