from app import ma


class ChatMessageSchema(ma.Schema):
    role = ma.String(required=True)
    content = ma.String(required=True)


class ChatMessagesSchema(ma.Schema):
    messages = ma.List(ma.Nested(ChatMessageSchema), required=True)


chat_messages_schema = ChatMessagesSchema()
