"""ANTLR4-based visitor for parsing TAIDL semantics into HLO generation code"""

from taidl.antlr4 import IDLV2Parser, IDLV2Visitor

from .template import generate_code, templates, init_templates


class OracleVisitor(IDLV2Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.instruction_lines = []
        init_templates()

        self.op_to_template = {
            'reshape': 'RESHAPE',
            'convert': 'CONVERT',
            'copy': 'COPY',
            'bitcast_convert': 'BITCAST_CONVERT',
            'exponential': 'EXP',
            'transpose': 'TRANSPOSE',
            'add': 'ADD',
            'subtract': 'SUBTRACT',
            'multiply': 'MULTIPLY',
            'divide': 'DIVIDE',
            'broadcast': 'BROADCAST',
            'maximum': 'MAXIMUM',
            'minimum': 'MINIMUM',
            'xor': 'XOR',
            'dot': 'DOT',
            'reduce_add': 'REDUCE_ADD',
            # Ops whose templates already existed but were never reachable
            # because the visitor had no mapping for them.
            'exp': 'EXP',
            'constant': 'CONSTANT',
            'reduce': 'REDUCE',
            'select_lt': 'SELECT_LT',
        }

        # Identity element for a generic `reduce`, keyed by the prefix of the
        # to_apply function. `reduce` may also pass init=<literal> explicitly.
        self.reduce_init = {'add': '0', 'max': '-inf', 'min': 'inf'}

    def visitModule(self, ctx: IDLV2Parser.ModuleContext):
        self.instruction_lines.append("Module:")
        return self.visitChildren(ctx)

    def visitInstruction(self, ctx: IDLV2Parser.InstructionContext):
        if ctx.ROOT():
            pass

        lhs_name = ctx.IDENTIFIER().getText().lstrip('%')
        result_type_info = self.visit(ctx.result_type())
        lhs_type, lhs_shape = result_type_info
        op_name = ctx.OPERATION().getText()

        if op_name == 'parameter':
            return (lhs_name, lhs_type, lhs_shape)

        operand_names = []
        if ctx.operands():
            operand_info = self.visit(ctx.operands())
            for operand_value, operand_type in operand_info:
                if operand_type == 'IDENTIFIER':
                    operand_names.append(operand_value.lstrip('%'))
                else:
                    operand_names.append(operand_value)

        attributes = {}
        if ctx.attributes():
            attribute_info = self.visit(ctx.attributes())
            for attr_name, attr_value, attr_type in attribute_info:
                if attr_type == 'BRACELIST':
                    values = [v['value'] for v in attr_value]
                    attributes[attr_name] = values
                else:
                    attributes[attr_name] = attr_value

        template_name = self.op_to_template[op_name]
        self._generate_from_template(template_name, op_name, lhs_name, lhs_type, lhs_shape,
                                     operand_names, attributes)

        self.instruction_lines.append(f"lvars['{lhs_name}'] = lhs_loc")

        return (lhs_name, lhs_type, lhs_shape)

    def _generate_from_template(self, template_name, op_name, lhs_name, lhs_type, lhs_shape,
                                operand_names, attributes):
        mapping = {
            "lhs": f'"{lhs_name}"',
            "type": f'"{lhs_type}"',
            "size": f'[{lhs_shape}]'
        }

        if template_name in ['RESHAPE', 'CONVERT', 'COPY', 'BITCAST_CONVERT', 'EXP']:
            if len(operand_names) > 0:
                mapping["in"] = f'"{operand_names[0]}"'

        elif template_name in ['ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'MAXIMUM', 'MINIMUM', 'XOR']:
            if len(operand_names) >= 2:
                mapping["A"] = f'"{operand_names[0]}"'
                mapping["B"] = f'"{operand_names[1]}"'

        elif template_name == 'BROADCAST':
            if len(operand_names) > 0:
                mapping["A"] = f'"{operand_names[0]}"'
            # Which output dimensions the operand's dimensions map onto. A
            # scalar operand broadcasts with dimensions={}; the template used to
            # ignore this attribute entirely and emit an undefined name.
            dims = attributes.get('dimensions')
            if dims is None:
                mapping["dims"] = ''
            elif isinstance(dims, list):
                mapping["dims"] = ', '.join(str(d) for d in dims)
            else:
                mapping["dims"] = str(dims)

        elif template_name == 'TRANSPOSE':
            if len(operand_names) > 0:
                mapping["in"] = f'"{operand_names[0]}"'
            if 'dimensions' in attributes:
                dims = attributes['dimensions']
                if isinstance(dims, list):
                    mapping["dims"] = ", ".join(str(d) for d in dims)
                else:
                    mapping["dims"] = str(dims)

        elif template_name == 'DOT':
            if len(operand_names) >= 2:
                mapping["A"] = f'"{operand_names[0]}"'
                mapping["B"] = f'"{operand_names[1]}"'

            def format_dims(value):
                if isinstance(value, list):
                    return '{' + ', '.join(str(v) for v in value) + '}'
                elif isinstance(value, str):
                    return value
                else:
                    return '{}'

            mapping["lb"] = format_dims(attributes.get('lhs_batch_dims', []))
            mapping["lc"] = format_dims(attributes.get('lhs_contracting_dims', [1]))
            mapping["rb"] = format_dims(attributes.get('rhs_batch_dims', []))
            mapping["rc"] = format_dims(attributes.get('rhs_contracting_dims', [0]))

        elif template_name == 'CONSTANT':
            # `constant` has no operands in HLO's sense -- the literal rides in
            # the operand slot: `%one = bf16[1] constant(1)`.
            assert len(operand_names) == 1, "constant takes exactly one literal"
            mapping["const"] = f'"{operand_names[0]}"'
            # A literal is a scalar, but the grammar's `shape` rule needs at
            # least one dimension, so there is no way to spell `TYPE[]`. Treat
            # the one spellable degenerate shape, `TYPE[1]`, as that scalar --
            # HLO only broadcasts from a scalar, never from a size-1 dimension.
            if lhs_shape.strip() == '1':
                mapping["size"] = '[]'

        elif template_name == 'SELECT_LT':
            assert len(operand_names) == 4, "select_lt takes (lhs, rhs, on_lt, on_ge)"
            for key, name in zip(('A', 'B', 'C', 'D'), operand_names):
                mapping[key] = f'"{name}"'

        elif template_name in ('REDUCE_ADD', 'REDUCE'):
            assert(len(operand_names) == 1)
            mapping["A"] = f'"{operand_names[0]}"'
            mapping["dims"] = self._format_dims(attributes.get('dimensions'))
            if template_name == 'REDUCE_ADD':
                to_apply = f'%add_{lhs_type}'
            else:
                to_apply = attributes.get('to_apply')
                assert to_apply, "reduce requires a to_apply=%<fn> attribute"
                if not to_apply.startswith('%'):
                    to_apply = '%' + to_apply
            mapping["to_apply"] = f'"{to_apply}"'
            # The reduction's identity element. Derived from the combiner's name
            # (add -> 0, max -> -inf, min -> inf) unless given explicitly.
            init = attributes.get('init')
            if init is None:
                combiner = to_apply.lstrip('%').split('_')[0]
                init = self.reduce_init.get(combiner)
                assert init is not None, (
                    f"cannot infer an identity for to_apply={to_apply}; "
                    f"pass init=<literal> explicitly")
            mapping["init"] = f'"{init}"'
            template_name = "REDUCE"

        template_code = generate_code(templates[template_name], mapping)

        for line in template_code.strip().split('\n'):
            self.instruction_lines.append(line)

    @staticmethod
    def _format_dims(dims):
        """Render a `dimensions=` attribute as an HLO brace list."""
        if dims is None:
            return '{}'
        if isinstance(dims, list):
            return '{' + ', '.join(str(d) for d in dims) + '}'
        return '{' + str(dims) + '}'

    def visitAttribute(self, ctx: IDLV2Parser.AttributeContext):
        attr_name = ctx.IDENTIFIER().getText()
        attr_value_info = self.visit(ctx.attributeValue())
        attr_value, attr_type = attr_value_info
        return (attr_name, attr_value, attr_type)

    def visitAttributes(self, ctx: IDLV2Parser.AttributesContext):
        attributes_list = []
        for attribute_ctx in ctx.attribute():
            attr_info = self.visit(attribute_ctx)
            attributes_list.append(attr_info)
        return attributes_list

    def visitAttributeValue(self, ctx: IDLV2Parser.AttributeValueContext):
        if ctx.braceList():
            brace_info = self.visit(ctx.braceList())
            return (brace_info, 'BRACELIST')
        elif ctx.value():
            value_ctx = ctx.value()
            value_text = value_ctx.getText()

            if value_ctx.INT():
                value_type = 'INT'
            elif value_ctx.IDENTIFIER():
                value_type = 'IDENTIFIER'
            elif value_ctx.EXPRESSION():
                value_type = 'EXPRESSION'
            else:
                value_type = 'UNKNOWN'

            return (value_text, value_type)
        else:
            return ("", "UNKNOWN")

    def visitBraceList(self, ctx: IDLV2Parser.BraceListContext):
        if ctx.value():
            values = []
            for value_ctx in ctx.value():
                value_text = value_ctx.getText()

                if value_ctx.INT():
                    value_type = 'INT'
                elif value_ctx.IDENTIFIER():
                    value_type = 'IDENTIFIER'
                elif value_ctx.EXPRESSION():
                    value_type = 'EXPRESSION'
                else:
                    value_type = 'UNKNOWN'

                values.append({'value': value_text, 'type': value_type})
            return values
        else:
            return []

    def visitResult_type(self, ctx: IDLV2Parser.Result_typeContext):
        shape_info = self.visit(ctx.shape())
        shape_dims, shape_type = shape_info
        return (shape_type, shape_dims)

    def visitShape(self, ctx: IDLV2Parser.ShapeContext):
        dims = ctx.getText()
        dims = dims.replace("`", "'")
        typ = ctx.parentCtx.TYPE().getText()
        return (dims, typ)

    def visitOperands(self, ctx: IDLV2Parser.OperandsContext):
        operand_list = []
        for operand_ctx in ctx.operand():
            operand_info = self.visit(operand_ctx)
            operand_list.append(operand_info)
        return operand_list

    def visitOperand(self, ctx: IDLV2Parser.OperandContext):
        value_ctx = ctx.value()
        value_text = value_ctx.getText()

        if value_ctx.INT():
            operand_type = 'INT'
        elif value_ctx.IDENTIFIER():
            operand_type = 'IDENTIFIER'
        elif value_ctx.EXPRESSION():
            operand_type = 'EXPRESSION'
        else:
            operand_type = 'UNKNOWN'

        return (value_text, operand_type)
