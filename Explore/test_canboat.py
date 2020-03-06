import struct
import json


# Trying to figure out if there is a bug in canboat analyzer.  The current symptom is that
# there is never a negative rate of turn received.

# There are the raw logs
raw = [
    '2019-10-19T03:38:04.639Z,2,127251,4,255,8,ff,9f,39,fb,ff,ff,ff,ff',
    '2019-10-19T03:38:04.639Z,3,127251,128,255,8,00,03,bb,f9,ff,00,ff,ff',
    '2019-10-19T03:38:04.640Z,2,127251,4,255,8,ff,f4,3d,fa,ff,ff,ff,ff',
    '2019-10-19T03:38:04.640Z,2,127251,4,255,8,ff,c7,c1,fa,ff,ff,ff,ff',
    '2019-10-19T03:38:04.640Z,3,127251,128,255,8,00,0d,93,f9,ff,00,ff,ff',
    '2019-10-19T03:38:04.640Z,2,127251,4,255,8,ff,9f,39,fb,ff,ff,ff,ff',
    '2019-10-19T03:38:04.641Z,2,127251,4,255,8,ff,9f,39,fb,ff,ff,ff,ff'
    ]

# These are the resulting outputs:
canboat_json = [
    '{"timestamp":"2019-10-19T03:38:04.639Z","prio":2,"src":4,"dst":255,"pgn":127251,"description":"Rate of Turn","fields":{"Rate":-0.00017}}',
    '{"timestamp":"2019-10-19T03:38:04.639Z","prio":3,"src":128,"dst":255,"pgn":127251,"description":"Rate of Turn","fields":{"SID":0,"Rate":-0.00045}}',
    '{"timestamp":"2019-10-19T03:38:04.640Z","prio":2,"src":4,"dst":255,"pgn":127251,"description":"Rate of Turn","fields":{"Rate":-0.00002}}',
    '{"timestamp":"2019-10-19T03:38:04.640Z","prio":2,"src":4,"dst":255,"pgn":127251,"description":"Rate of Turn","fields":{"Rate":-0.00010}}',
    '{"timestamp":"2019-10-19T03:38:04.640Z","prio":3,"src":128,"dst":255,"pgn":127251,"description":"Rate of Turn","fields":{"SID":0,"Rate":-0.00044}}',
    '{"timestamp":"2019-10-19T03:38:04.640Z","prio":2,"src":4,"dst":255,"pgn":127251,"description":"Rate of Turn","fields":{"Rate":-0.00017}}',
    '{"timestamp":"2019-10-19T03:38:04.641Z","prio":2,"src":4,"dst":255,"pgn":127251,"description":"Rate of Turn","fields":{"Rate":-0.00017}}',
    '{"timestamp":"2019-10-19T03:38:04.641Z","prio":3,"src":128,"dst":255,"pgn":127251,"description":"Rate of Turn","fields":{"SID":0,"Rate":-0.00015}}',
    '{"timestamp":"2019-10-19T03:38:04.641Z","prio":2,"src":4,"dst":255,"pgn":127251,"description":"Rate of Turn","fields":{"Rate":-0.00025}}',
    '{"timestamp":"2019-10-19T03:38:04.642Z","prio":2,"src":4,"dst":255,"pgn":127251,"description":"Rate of Turn","fields":{"Rate":-0.00025}}'
]

# The PGN is 127251d
#
# Looks like 5 bytes.  An SID field (which groups messages) and 4 bytes of rate.
# {
#   "PGN":127251,
#   "Id":"rateOfTurn",
#   "Description":"Rate of Turn",
#   "Complete":true,
#   "Length":5,
#   "RepeatingFields":0,
#   "Fields":[
#     {
#       "Order":1,
#       "Id":"sid",
#       "Name":"SID",
#       "BitLength":8,
#       "BitOffset":0,
#       "BitStart":0,
#       "Signed":false},
#     {
#       "Order":2,
#       "Id":"rate",
#       "Name":"Rate",
#       "BitLength":32,
#       "BitOffset":8,
#       "BitStart":0,
#       "Units":"rad/s",
#       "Resolution":3.125e-08,
#       "Signed":true}]},

# Are these bits interpretted correctly:

# 2019-10-19T03:38:32.789Z 2   4 255 127251 Rate of Turn:
# decode SID offset=0 startBit=0 bits=8 bytes=1:  SID = Unknown
# decode Rate offset=1 startBit=0 bits=32 bytes=4:; Rate = 0.00661522 rad/s
# 2019-10-19T03:38:32.789Z 2 004 255 127251 : ff e7 3a 03 00 ff ff ff
# 2019-10-19T03:38:32.845Z 2   4 255 127251 Rate of Turn:
# decode SID offset=0 startBit=0 bits=8 bytes=1:  SID = Unknown
# decode Rate offset=1 startBit=0 bits=32 bytes=4:; Rate = 0.00450603 rad/s
# 2019-10-19T03:38:32.845Z 2 004 255 127251 : ff 41 33 02 00 ff ff ff
# 2019-10-19T03:38:32.895Z 2   4 255 127251 Rate of Turn:
# decode SID offset=0 startBit=0 bits=8 bytes=1:  SID = Unknown
# decode Rate offset=1 startBit=0 bits=32 bytes=4:; Rate = 0.00345141 rad/s
# 2019-10-19T03:38:32.895Z 2 004 255 127251 : ff 6d af 01 00 ff ff ff
# 2019-10-19T03:38:32.927Z 2   4 255 127251 Rate of Turn:
# decode SID offset=0 startBit=0 bits=8 bytes=1:  SID = Unknown
# decode Rate offset=1 startBit=0 bits=32 bytes=4:; Rate = 0.00143809 rad/s
# 2019-10-19T03:38:32.927Z 2 004 255 127251 : ff c3 b3 00 00 ff ff ff
# 2019-10-19T03:38:32.967Z 2   4 255 127251 Rate of Turn:
# decode SID offset=0 startBit=0 bits=8 bytes=1:  SID = Unknown
# decode Rate offset=1 startBit=0 bits=32 bytes=4:; Rate = -0.00000016 rad/s
# 2019-10-19T03:38:32.967Z 2 004 255 127251 : ff 19 b8 ff ff ff ff ff
# 2019-10-19T03:38:33.029Z 2   4 255 127251 Rate of Turn:
# decode SID offset=0 startBit=0 bits=8 bytes=1:  SID = Unknown
# decode Rate offset=1 startBit=0 bits=32 bytes=4:; Rate = -0.00000016 rad/s
# 2019-10-19T03:38:33.029Z 2 004 255 127251 : ff 19 b8 ff ff ff ff ff
# 2019-10-19T03:38:33.100Z 2   4 255 127251 Rate of Turn:
# decode SID offset=0 startBit=0 bits=8 bytes=1:  SID = Unknown
# decode Rate offset=1 startBit=0 bits=32 bytes=4:; Rate = -0.00000003 rad/s
# 2019-10-19T03:38:33.100Z 2 004 255 127251 : ff 46 34 ff ff ff ff ff

examples = [
    dict(Rate = 0.00661522, byte_string = "e7 3a 03 00 ff ff ff"),
    dict(Rate = 0.00450603, byte_string = "41 33 02 00 ff ff ff"),
    dict(Rate = 0.00345141, byte_string = "6d af 01 00 ff ff ff"),
    dict(Rate = 0.00143809, byte_string = "c3 b3 00 00 ff ff ff"),
    dict(Rate = -0.00000016, byte_string = "19 b8 ff ff ff ff ff"),
    dict(Rate = -0.00000016, byte_string = "19 b8 ff ff ff ff ff"),
    dict(Rate = -0.00000003, byte_string = "46 34 ff ff ff ff ff")
]

test_log = '/Users/viola/canlogs/OldLogs/test.log'
test_json = '/Users/viola/canlogs/OldLogs/test.json'

with open(test_log) as flog:
    log_lines = [l.strip() for l in flog.readlines()]

with open(test_json) as fjson:
    json_lines = [json.loads(l) for l in fjson.readlines()]

m = 1e-6 / 32.0

for l, j in zip(log_lines, json_lines):
    print()
    print(l)
    print(j)
    byte_sequence = l.strip().split(',')[-7:]
    canboat_rate = j['fields']['Rate']
    hex_string = ''.join(byte_sequence[:4])

    bb = bytes.fromhex(hex_string)
    val = struct.unpack('<i', bb)[0]
    rate = val * m
    match = (canboat_rate - rate) < 0.000001
    if match:
        print(f'    Values match {canboat_rate:.5f}, {rate:.5f}')
    else:
        print(f'*** Values don\'t match {canboat_rate:.5f}, {rate:.5f}')


