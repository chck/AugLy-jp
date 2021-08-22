class Action:
    INSERT = 'insert'
    SUBSTITUTE = 'substitute'

    @staticmethod
    def getall():
        return [Action.INSERT, Action.SUBSTITUTE]
