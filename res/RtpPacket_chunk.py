import sys
import time

HEADER_SIZE = 20

RTP_CURRENT_VERSION = 1
RTP_PADDLING_TRUE = 1
RTP_PADDLING_FALSE = 0
RTP_EXTENSION_TRUE = 1
RTP_EXTENSION_FALSE = 0
RTP_CC = 0
RTP_MARKER_TRUE = 1
RTP_MARKER_FALSE = 0
RTP_TYPE_JPEG = 26
RTP_SSRC = 0


class RtpPacket:
    header = bytearray(HEADER_SIZE)

    def __init__(self):
        pass

    def encode(self, seqnum, marker, timestamp, device_ID, video_ID, chunk_ID, step_index, payload, episode, start_send):
        return self._encode(RTP_CURRENT_VERSION,
                            RTP_PADDLING_FALSE,
                            RTP_EXTENSION_FALSE,
                            RTP_CC,
                            seqnum,
                            marker,
                            RTP_TYPE_JPEG,
                            RTP_SSRC,
                            timestamp,
                            device_ID,
                            video_ID,
                            chunk_ID,
                            step_index,
                            payload,
                            episode,
                            start_send)

    def _encode(self, version, padding, extension, cc, seqnum, marker, pt, ssrc, timestamp, device_ID, video_ID, chunk_ID, step_index, payload, episode, start_send):
        timestamp_s = int(timestamp)
        timestamp_ms = int(timestamp * 100 - timestamp_s * 100)

        send_timestamp_s = int(start_send)
        send_timestamp_ms = int(start_send * 100 - send_timestamp_s * 100)
        header = bytearray(HEADER_SIZE)
        # Fill the header bytearray with RTP header fields
        header[0] = (version << 6) | (padding << 5) | (extension << 4) | cc
        # header[1] = (marker << 7) | pt
        # 0 for new image, 1 for normal image, 2 for end image
        header[1] = marker
        header[2] = (seqnum >> 8) & 255  # upper bits
        header[3] = seqnum & 255
        header[4] = timestamp_s >> 24 & 255
        header[5] = timestamp_s >> 16 & 255
        header[6] = timestamp_s >> 8 & 255
        header[7] = timestamp_s & 255
        ## millisecond
        header[8] = timestamp_ms & 255
        header[9] = device_ID & 255
        header[10] = video_ID & 255
        header[11] = chunk_ID & 255
        header[12] = step_index >> 8 & 255  # high byte
        header[13] = step_index & 255  # low byte
        header[14] = episode & 255

        header[15] = send_timestamp_s >> 24 & 255
        header[16] = send_timestamp_s >> 16 & 255
        header[17] = send_timestamp_s >> 8 & 255
        header[18] = send_timestamp_s & 255
        ## millisecond
        header[19] = send_timestamp_ms & 255

        self.header = header
        ##print('header.type.len=', type(header), len(header))
        # Get the payload from the argument
        self.payload = payload

    def decode(self, byteStream):
        """Decode the RTP packet."""
        self.header = bytearray(byteStream[:HEADER_SIZE])
        self.payload = byteStream[HEADER_SIZE:]

    def version(self):
        """Return RTP version."""
        return int(self.header[0] >> 6)

    def seqNum(self):
        """Return sequence (frame) number."""
        seqNum = self.header[2] << 8 | self.header[3]
        return int(seqNum)

    def setTimestamp(self, timestamp):
        header_tmp = self.header
        timestamp_s = int(timestamp)
        timestamp_ms = int(timestamp * 100 - timestamp_s * 100)
        header = bytearray(HEADER_SIZE)
        # Fill the header bytearray with RTP header fields
        header_tmp[4] = timestamp_s >> 24 & 255
        header_tmp[5] = timestamp_s >> 16 & 255
        header_tmp[6] = timestamp_s >> 8 & 255
        header_tmp[7] = timestamp_s & 255
        header_tmp[8] = timestamp_ms & 255

        self.header = header_tmp

    def getTimestamp(self):
        """Return timestamp."""
        timestamp = self.header[4] << 24 | self.header[5] << 16 | self.header[6] << 8 | self.header[7]
        return float(timestamp) + self.header[8] / 100

    def getSendTimestamp(self):
        """Return send timestamp."""
        timestamp = self.header[15] << 24 | self.header[16] << 16 | self.header[17] << 8 | self.header[18]
        return float(timestamp) + self.header[19] / 100

    def Marker(self):
        '''
        Return Marker Type
        0 for new image, 1 for the same image, 2 for end of image
        '''
        marker = self.header[1]
        return marker

    # def payloadType(self):
    #     """Return payload type."""
    #     pt = self.header[1] & 127
    #     return int(pt)

    # def setPayLoad(self, payload):
    #     """Set the payload"""
    #     self.payload = payload
    #
    def getPayload(self):
        """Return payload."""
        return self.payload

    def getDeviceID(self):
        '''
        Return device_ID
        '''
        device_ID = self.header[9]
        return device_ID

    def getVideoID(self):
        '''
        Return video_ID
        '''
        video_ID = self.header[10]
        return video_ID

    def getChunkID(self):
        '''
        Return chunk_ID
        '''
        chunk_ID = self.header[11]
        return chunk_ID

    def getStepIndex(self):
        '''
        Return step_index
        '''
        step_index = self.header[12] << 8 | self.header[13]
        return step_index

    def getEpisode(self):
        '''
        Return step_index
        '''
        episode = self.header[14]
        return episode

    def getPacket(self):
        """Return RTP packet."""
        return self.header + self.payload


def main():
   print((1 << 7))


if __name__ == "__main__":
    main()