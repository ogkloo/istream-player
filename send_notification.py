import zmq
import argparse

from time import sleep

def main(msg, timeout, port, ack_port):
    ctx = zmq.Context()
    publisher = ctx.socket(zmq.PUB)
    publisher.bind(f"tcp://127.0.0.1:{port}")

    ack_recvr = ctx.socket(zmq.SUB)
    ack_recvr.connect(f"tcp://127.0.0.1:{ack_port}")
    ack_recvr.setsockopt_string(zmq.SUBSCRIBE, "ack")


    sleep(0.5)
    publisher.send_string(f"evs {msg}")

    if timeout < 0:
        print("Sent. Waiting...")
        resend_attempts = 0
        sleep(0.5)
        while resend_attempts < timeout:
            try:
                ack = ack_recvr.recv_string(flags=zmq.NOBLOCK)
                print(f"Got ack {ack}")
                publisher.close()
                ack_recvr.close()
                ctx.term()
                return
            except zmq.Again:
                print("Resending")
                sleep(0.5)
                publisher.send_string(f"evs {msg}")
            resend_attempts += 1

        print("connection timed out")

    publisher.close()
    ack_recvr.close()
    ctx.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sends an event message over ZMQ.')
    parser.add_argument("-m", "--message", type=str, 
                        help="Message to send. Should be JSON.", 
                        required=True)
    parser.add_argument("-p", "--send_port", type=int, 
                        help="Port to send message on.",
                        default=5555)
    parser.add_argument("-a", "--ack_port", type=int, 
                        help="Port to receive ack on.",
                        default=5554)
    parser.add_argument("-t", "--timeout", type=int, 
                        help="Number of resends to attempt. 0 will not check for an ACK.",
                        default=0)

    args = parser.parse_args()

    message: str = args.message
    send_port: int = args.send_port
    ack_port: int = args.ack_port
    timeout: int = args.timeout

    main(message, timeout, send_port, ack_port)