# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/recognizer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16proto/recognizer.proto\x12\nrecognizer\"#\n\x08ImageURL\x12\x0b\n\x03url\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\t\"\x1b\n\x0bImageBase64\x12\x0c\n\x04\x62\x61se\x18\x01 \x01(\t\"\x16\n\x07\x41\x63\x63ount\x12\x0b\n\x03msg\x18\x01 \x01(\t\"\x15\n\x06Notice\x12\x0b\n\x03res\x18\x01 \x01(\t2N\n\x0cImageService\x12>\n\x0eImageReqBase64\x12\x17.recognizer.ImageBase64\x1a\x13.recognizer.Account2H\n\rImageRegistor\x12\x37\n\x0bImageReqURL\x12\x14.recognizer.ImageURL\x1a\x12.recognizer.NoticeB\nZ\x08pkg/grpcb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.recognizer_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\010pkg/grpc'
  _globals['_IMAGEURL']._serialized_start=38
  _globals['_IMAGEURL']._serialized_end=73
  _globals['_IMAGEBASE64']._serialized_start=75
  _globals['_IMAGEBASE64']._serialized_end=102
  _globals['_ACCOUNT']._serialized_start=104
  _globals['_ACCOUNT']._serialized_end=126
  _globals['_NOTICE']._serialized_start=128
  _globals['_NOTICE']._serialized_end=149
  _globals['_IMAGESERVICE']._serialized_start=151
  _globals['_IMAGESERVICE']._serialized_end=229
  _globals['_IMAGEREGISTOR']._serialized_start=231
  _globals['_IMAGEREGISTOR']._serialized_end=303
# @@protoc_insertion_point(module_scope)
