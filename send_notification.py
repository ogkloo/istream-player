import zmq
import argparse

from time import sleep

def main(msg, port=5555, ack_port=5554):
    ctx = zmq.Context()
    publisher = ctx.socket(zmq.PUB)
    publisher.bind(f"tcp://localhost:{port}")

    ack_recvr = ctx.socket(zmq.SUB)
    ack_recvr.connect(f"tcp://localhost:{ack_port}")
    ack_recvr.setsockopt_string(zmq.SUBSCRIBE, "ack")

    print("sending message")

    sleep(0.1)
    publisher.send_string(f"evs {msg}")

    while True:
        try:
            sleep(1)
            ack = ack_recvr.recv_string(flags=zmq.NOBLOCK)
            print(f"Got ack {ack}")
            return
        except zmq.Again:
            # No message, continue with other tasks
            sleep(0.5)
            print("Resending")
            publisher.send_string(f"evs {msg}")


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

    args = parser.parse_args()

    message: str = args.message
    send_port: int = args.send_port
    ack_port: int = args.ack_port

    main(message, send_port, ack_port)