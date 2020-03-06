
import socket
import pynmea2
import pickle
import itertools as it

# Code to read the NMEA 0183 strings being sent over WIFI on the boat.  Note, this is not
# NEMA 2K.  It appears that the Vulcan reads the 2K and converts it to 0183 (or at least
# some of it).

# Notes:
#
# The IIXDR message has multiple fields, only the first is currently parse...  need to extend.
#
# The data I have is not showing *any* wind (always 0.0).  Not sure if this is a bug, or
# if it was actually true.


# References
#
# https://github.com/Knio/pynmea2
#
# - https://opencpn.org/wiki/dokuwiki/doku.php?id=opencpn:opencpn_user_manual:advanced_features:nmea_sentences
# 
# - https://gpsd.gitlab.io/gpsd/NMEA.html
#
# - http://www.nuovamarea.net/blog/wimwv
# - http://www.nuovamarea.net/blog/WIMWD
#
# Other References
#
# https://docs.python.org/3/howto/sockets.html


SAVE_FILE = "/Users/viola/canlogs/nmea_chunks.pickle"

def read_boat_data(save_to_file=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("192.168.0.58", 10110))

    chunks = []
    for i in range(1000):
        # Just read it as fast as it comes.
        mm = s.recv(2048)
        print(i, len(mm))
        chunks.append(mm)

    if save_to_file:
        with open(save_to_file, 'wb') as fs:
            pickle.dump(chunks, fs)

    return chunks

def read_cached_data(from_file):
    with open(from_file, 'rb') as fs:
        chunks = pickle.load(fs)
    return chunks


chunks = read_cached_data(SAVE_FILE)


def decode_messages(chunks):
    # Decode binary chunkds into ascii string
    ascii_string = ''
    for ch in chunks:
        ascii_string += ch.decode('ascii')

    # seperate into raw messages
    raw_nmea_messages = ascii_string.split('\r\n')

    dd = {}
    for i, raw_message in it.islice(zip(it.count(), raw_nmea_messages), 10, 100000):
        # print(i, raw_message)
        try:
            match = pynmea2.NMEASentence.sentence_re.match(raw_message)
            if not match:
                print('*** could not parse data', line)
            else:
                # pylint: disable=bad-whitespace
                nmea_str        = match.group('nmea_str')
                data_str        = match.group('data')
                checksum        = match.group('checksum')
                sentence_type   = match.group('sentence_type').upper()
                data            = data_str.split(',')
                dd[sentence_type] = None
                # print(sentence_type, data_str)
                msg = pynmea2.parse(raw_message)
                dd[sentence_type] = msg.__repr__()
                if sentence_type == 'WIMWV,':
                    print(nmea_str)
                    print(msg.__repr__())
        except:
            pass
