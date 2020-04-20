import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-inputFilePath',          type=str,   default='hw_wordsim.py',  help="Path to your submission file.")
args = parser.parse_args()

START_TEXT = r"# Enter your system description in this cell"
END_TEXT   = r"# Please do not remove this comment"
def extractOriginalSystem(fileText):
  regEX = r"(?s)" + START_TEXT + ".*?" + END_TEXT
  OSYS_RE = re.compile(regEX)
  return OSYS_RE.findall(fileText)


def main():
  with open(args.inputFilePath, "r") as inFile:
    fileText = inFile.read()
  results = extractOriginalSystem(fileText)
  if len(results) > 0:
    print(results[0])

main()