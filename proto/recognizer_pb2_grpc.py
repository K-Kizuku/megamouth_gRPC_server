# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import recognizer_pb2 as recognizer__pb2


class ImageServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ImageReq = channel.unary_unary(
                '/recognizer.ImageService/ImageReq',
                request_serializer=recognizer__pb2.ImageURL.SerializeToString,
                response_deserializer=recognizer__pb2.Account.FromString,
                )


class ImageServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ImageReq(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ImageServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ImageReq': grpc.unary_unary_rpc_method_handler(
                    servicer.ImageReq,
                    request_deserializer=recognizer__pb2.ImageURL.FromString,
                    response_serializer=recognizer__pb2.Account.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'recognizer.ImageService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ImageService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ImageReq(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/recognizer.ImageService/ImageReq',
            recognizer__pb2.ImageURL.SerializeToString,
            recognizer__pb2.Account.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)