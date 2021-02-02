"""Utilities for working with subject data."""

import os
import re
import time
import subprocess as sub
from datetime import datetime
from glob import glob
from argparse import ArgumentParser


def imname(filepath):
    """Get the name of an .nii.gz file."""
    return os.path.basename(filepath.split(".nii.gz")[0])


def impath(*args):
    """Construct the path to an .nii.gz file."""
    p = os.path.join(*args)
    if not p.endswith(".nii.gz"):
        p += ".nii.gz"
    return p


class SubjParser(ArgumentParser):
    """Class for parsing standard arguments."""

    def __init__(self, include_log=True, raw=False, **init_args):
        if raw:
            from argparse import RawTextHelpFormatter

            ArgumentParser.__init__(
                self, formatter_class=RawTextHelpFormatter, **init_args
            )
        else:
            ArgumentParser.__init__(self, **init_args)
        self.add_argument(
            "subject", type=str, help="name of subject directory in study directory"
        )
        if "STUDYDIR" in os.environ:
            study_dir = os.environ["STUDYDIR"]
        else:
            study_dir = None

        s = """path to main study directory; if not set, value of STUDYDIR
environment variable will be used"""
        self.add_argument("--study-dir", type=str, default=study_dir, help=s)
        if include_log:
            self.add_argument(
                "--dry-run",
                help="display commands without executing",
                default=False,
                action="store_true",
            )
            self.add_argument(
                "--clean-logs",
                help="remove existing similar logs",
                default=False,
                action="store_true",
            )


class SubjLog:
    """Class for logging subject processing."""

    def __init__(
        self, subject, base, main=None, rm_existing=False, dry_run=False, study_dir=None
    ):

        self.subject = subject
        self.name = base
        self.dry_run = dry_run
        self.main_file = None
        self.start_time = None

        # set the log file
        if study_dir is None:
            study_dir = os.environ["STUDYDIR"]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = base + "_" + timestamp + ".log"
        log_dir = os.path.join(study_dir, subject, "logs")
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_file = os.path.join(log_dir, filename)
        if main is not None:
            filename = main + ".log"
            self.main_file = os.path.join(log_dir, filename)

        if rm_existing:
            # clear logs that match the supplied base
            existing = glob(os.path.join(log_dir, "%s_*.log" % base))
            for filepath in existing:
                os.remove(filepath)
        self.log_file = log_file

    def timestamp(self):
        """Get a timestamp with standard formatting for a log."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def start(self):
        """Start logging."""

        if self.dry_run:
            return

        # print a header
        msg = "Starting %s for %s at: %s\n" % (
            self.name,
            self.subject,
            self.timestamp(),
        )
        self.start_time = time.time()
        self.write(msg, wrap=False)
        if self.main_file:
            self.write(msg, wrap=False, main_log=True)

    def finish(self):
        """Finish and close the log."""

        if self.dry_run:
            return

        finish = time.time() - self.start_time
        msg = "Finished %s for %s at: %s\n" % (
            self.name,
            self.subject,
            self.timestamp(),
        )
        self.write("\n" + msg, wrap=False)
        self.write("Took %d s." % finish, wrap=False)
        if self.main_file:
            self.write(msg, wrap=False, main_log=True)

    def run(self, cmd):
        """Run a command with input and output logging."""

        print(cmd)
        if self.dry_run:
            return

        # open log file and print command to run
        outfile = open(self.log_file, "a")
        outfile.write("\n" + cmd + "\n")
        outfile.close()

        # actually running the command
        p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE, shell=True)
        output, errors = p.communicate()
        outfile = open(self.log_file, "a")
        if output:
            print(output)
            outfile.write(output)
        if errors:
            outfile.write("ERROR: " + errors)
            print("%s: ERROR: " % self.subject + errors)
        outfile.close()

    def write(self, message, wrap=True, main_log=False):
        """Write a message to the log."""

        if self.dry_run:
            print(message)
            return

        if main_log:
            outfile = open(self.main_file, "a")
        else:
            outfile = open(self.log_file, "a")
        if wrap:
            outfile.write("\nMESSAGE: " + message + "\n")
        else:
            outfile.write(message)
        outfile.close()


class SubjPath:
    """Information about subject directory structure."""

    def __init__(self, subject, study_dir=None):

        self.subject = subject
        if study_dir is not None:
            self.study_dir = study_dir
        else:
            self.study_dir = os.environ["STUDYDIR"]
        self.subj_dir = os.path.join(self.study_dir, self.subject)
        self.dirnames = [
            "anatomy",
            "behav",
            "BOLD",
            "DTI",
            "fieldmap",
            "logs",
            "model",
            "raw",
        ]
        self.d = dict()
        self.d["base"] = self.subj_dir
        for dirname in self.dirnames:
            self.d[dirname.lower()] = os.path.join(self.subj_dir, dirname)

    def __str__(self):
        return "Subject %s (%s)" % (self.subject, self.subj_dir)

    def make_std_dirs(self):
        """Make the set of standard subject directories."""

        if not os.path.exists(self.d["base"]):
            os.mkdir(self.d["base"])
        for std in self.dirnames:
            name = std.lower()
            if not os.path.exists(self.d[name]):
                os.mkdir(self.d[name])

    def path(self, std, *args):
        """Get the path to file or directory within a standard directory."""
        if std in self.d:
            fulldir = os.path.join(self.d[std.lower()], *args)
        else:
            fulldir = os.path.join(self.subj_dir, std, *args)
        return fulldir

    def image_path(self, std, *args):
        """Get the path to an image file for a subject."""
        return impath(self.path(std, *args))

    def proj_path(self, *args):
        """Path to a file within the code project."""
        proj_dir = os.path.dirname(os.path.dirname(__file__))
        fulldir = os.path.join(proj_dir, *args)
        return fulldir

    def glob(self, std, *args):
        """Get all files or directories matching a pattern with *."""
        paths = glob(os.path.join(self.d[std.lower()], *args))
        return paths

    def match_dirs(self, main_dir, dir_pattern):
        """Get directories matching a regular expression."""
        dirs = os.listdir(main_dir)
        test = re.compile(dir_pattern)
        match = []
        for d in dirs:
            full = os.path.join(main_dir, d)
            if test.match(d) and os.path.isdir(full):
                match.append(full)
        match.sort()
        return match

    def init_log(self, base, main, args):
        """Initialize a log for processing this subject."""
        log = SubjLog(
            self.subject,
            base,
            main,
            rm_existing=args.clean_logs,
            dry_run=args.dry_run,
            study_dir=args.study_dir,
        )
        return log
