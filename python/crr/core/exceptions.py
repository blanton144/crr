# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import print_function, division, absolute_import


class CrrError(Exception):
    """A custom core Crr exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(CrrError, self).__init__(message)


class CrrNotImplemented(CrrError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(CrrNotImplemented, self).__init__(message)


class CrrAPIError(CrrError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Crr API'
        else:
            message = 'Http response error from Crr API. {0}'.format(message)

        super(CrrAPIError, self).__init__(message)


class CrrApiAuthError(CrrAPIError):
    """A custom exception for API authentication errors"""
    pass


class CrrMissingDependency(CrrError):
    """A custom exception for missing dependencies."""
    pass


class CrrWarning(Warning):
    """Base warning for Crr."""


class CrrUserWarning(UserWarning, CrrWarning):
    """The primary warning class."""
    pass


class CrrSkippedTestWarning(CrrUserWarning):
    """A warning for when a test is skipped."""
    pass


class CrrDeprecationWarning(CrrUserWarning):
    """A warning for deprecated features."""
    pass
