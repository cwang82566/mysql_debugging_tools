from __future__ import print_function # python2.X support
import re

#
# Some utility functions
#
def std_vector_to_list(std_vector):
    """convert std::vector to list of python"""
    out_list = []
    value_reference = std_vector['_M_impl']['_M_start']
    while value_reference != std_vector['_M_impl']['_M_finish']:
        out_list.append(value_reference.dereference())
        value_reference += 1

    return out_list

def std_deque_to_list(std_deque):
    """convert std::deque to list of python"""
    out_list = []
    elttype = std_deque.type.template_argument(0)
    size = elttype.sizeof
    if size < 512:
        buffer_size = int (512 / size)
    else:
        buffer_size = 1
    
    start = std_deque['_M_impl']['_M_start']['_M_cur']
    end = std_deque['_M_impl']['_M_start']['_M_last']
    node = std_deque['_M_impl']['_M_start']['_M_node']
    last = std_deque['_M_impl']['_M_finish']['_M_cur']

    p = start
    while p != last:
        out_list.append(p.dereference())
        p += 1
        if p != end:
            continue
        node += 1
        p = node[0]
        end = p + buffer_size

    return out_list

def mem_root_deque_to_list(deque):
    """convert mem_root_deque to list of python"""
    out_list = []
    elttype = deque.type.template_argument(0)
    elements = deque['block_elements']
    start = deque['m_begin_idx']
    end = deque['m_end_idx']
    blocks = deque['m_blocks']

    p = start
    while p != end:
        elt = blocks[p / elements]['elements'][p % elements]
        out_list.append(elt)
        p += 1

    return out_list

def gdb_set_convenience_variable(var_name, var):
    """set convenience variable in python script is supported by gdb 8.0
or above, this for lower gdb version support

    """
    pass

def get_convenince_name():
    return ''

#
# Global variables or version related functions definations
#
if hasattr(gdb, 'set_convenience_variable'):
    convenience_name_firstchar = 'a'
    convenience_name_sequence = [convenience_name_firstchar]

    def generate_convenince_name():
        global convenience_name_sequence
        convenience_name_maxlen = 2

        cname = ''.join(convenience_name_sequence)
        cnlen = len(convenience_name_sequence)
        for i, c in reversed(list(enumerate(convenience_name_sequence))):
            if c == 'z':
                continue
            convenience_name_sequence[i] = chr(ord(c) + 1)
            for j in range(i + 1, cnlen):
                convenience_name_sequence[j] = convenience_name_firstchar
            break
        else:
            convenience_name_sequence = [convenience_name_firstchar] * \
                (1 if cnlen == convenience_name_maxlen else (cnlen + 1))

        return cname

    def gdb_set_convenience_variable(var_name, var):
        gdb.set_convenience_variable(var_name, var)

    def get_convenince_name():
        return generate_convenince_name()

#
# Some convenience variables for easy debug because they are macros
#
gdb_set_convenience_variable('MAX_TABLES', gdb.parse_and_eval('sizeof(unsigned long long) * 8 - 3'))
gdb_set_convenience_variable('INNER_TABLE_BIT', gdb.parse_and_eval('((unsigned long long)1) << ($MAX_TABLES + 0)'))
gdb_set_convenience_variable('OUTER_REF_TABLE_BIT', gdb.parse_and_eval('((unsigned long long)1) << ($MAX_TABLES + 1)'))
gdb_set_convenience_variable('RAND_TABLE_BIT', gdb.parse_and_eval('((unsigned long long)1) << ($MAX_TABLES + 2)'))
gdb_set_convenience_variable('PSEUDO_TABLE_BITS', gdb.parse_and_eval('($INNER_TABLE_BIT | $OUTER_REF_TABLE_BIT | $RAND_TABLE_BIT)'))

# Define a mysql command prefix for all mysql related command
gdb.Command('mysql', gdb.COMMAND_DATA, prefix=True)

#
# Commands start here
#

#
# threads overview/search for mysql
#
def gdb_threads():
    if hasattr(gdb, 'selected_inferior'):
        threads = gdb.selected_inferior().threads()
    else:
        threads = gdb.inferiors()[0].threads()
    return threads

def pretty_frame_name(frame_name):
    """omit some stdc++ stacks"""
    pretty_names = (
        ('std::__invoke_impl', ''),
        ('std::__invoke', ''),
        ('std::_Bind', ''),
        ('Runnable::operator()', ''),
        ('std::thread::_Invoker', ''),
        ('std::thread::_State_impl', 'std::thread'),
        ('std::this_thread::sleep_for', 'std..sleep_for'))

    for templ, val in pretty_names:
        if frame_name.startswith(templ):
            return val

    return frame_name

def brief_backtrace(filter_threads):
    frames = ''
    frame = gdb.newest_frame() if hasattr(gdb, 'newest_frame') else gdb.selected_frame()
    while frame is not None:
        frame_name = frame.name() if frame.name() is not None else '??'
        if filter_threads is not None and frame_name in filter_threads:
            return None
        frame_name = pretty_frame_name(frame_name)
        if frame_name:
            frames += frame_name + ','
        frame = frame.older()
    frames = frames[:-1]
    return frames

class ThreadSearch(gdb.Command):
    """find threads given a regex which matchs thread name, parameter name or value"""

    def __init__ (self):
        super (ThreadSearch, self).__init__ ("thread search", gdb.COMMAND_OBSCURE)

    def invoke (self, arg, from_tty):
        pattern = re.compile(arg)
        threads = gdb_threads()
        old_thread = gdb.selected_thread()
        for thr in threads:
            thr.switch()
            backtrace = gdb.execute('bt', False, True)
            matched_frames = [fr for fr in backtrace.split('\n') if pattern.search(fr) is not None]
            if matched_frames:
                print(thr.num, brief_backtrace(None))

        old_thread.switch()

ThreadSearch()

class ThreadOverview(gdb.Command):
    """print threads overview, display all frames in one line and function name only for each frame"""
    # filter Innodb backgroud workers
    filter_threads = (
        # Innodb backgroud threads
        'log_closer',
        'buf_flush_page_coordinator_thread',
        'log_writer',
        'log_flusher',
        'log_write_notifier',
        'log_flush_notifier',
        'log_checkpointer',
        'lock_wait_timeout_thread',
        'srv_error_monitor_thread',
        'srv_monitor_thread',
        'buf_resize_thread',
        'buf_dump_thread',
        'dict_stats_thread',
        'fts_optimize_thread',
        'srv_purge_coordinator_thread',
        'srv_worker_thread',
        'srv_master_thread',
        'io_handler_thread',
        'event_scheduler_thread',
        'compress_gtid_table',
        'ngs::Scheduler_dynamic::worker_proxy'
        )
    def __init__ (self):
        super (ThreadOverview, self).__init__ ("thread overview", gdb.COMMAND_OBSCURE)

    def invoke (self, arg, from_tty):
        threads = gdb_threads()
        old_thread = gdb.selected_thread()
        thr_dict = {}
        for thr in threads:
            thr.switch()
            bframes = brief_backtrace(self.filter_threads)
            if bframes is None:
                continue
            if bframes in thr_dict:
                thr_dict[bframes].append(thr.num)
            else:
                thr_dict[bframes] = [thr.num,]
        thr_ow = [(v,k) for k,v in thr_dict.items()]
        thr_ow.sort(key = lambda l:len(l[0]), reverse=True)
        for nums_thr,funcs in thr_ow:
           print(','.join([str(i) for i in nums_thr]), funcs)
        old_thread.switch()

ThreadOverview()

class TreeWalker(object):
    """A base class for tree traverse"""

    def __init__(self):
        self.level_graph = []
        self.var_index = 0
        self.cname_prefix = None
        self.current_level = 0

    def reset(self):
        self.level_graph = []
        self.var_index = 0
        self.cname_prefix = get_convenince_name()

    def walk(self, expr):
        self.reset()
        self.do_walk(expr, 0)

    def do_walk(self, expr, level):
        item_show_info = ''
        expr_typed = None
        expr_nodetype = None
        if not isinstance(expr, str):
            expr_typed = expr.dynamic_type
            expr_casted = expr.cast(expr_typed)
            try:
                expr_nodetype = expr.type.template_argument(0)
                if expr_nodetype.code != gdb.TYPE_CODE_PTR:
                    expr_nodetype = expr.type.template_argument(0).pointer()
            except (gdb.error, RuntimeError):
                expr_nodetype = None
                pass
        else:
            item_show_info = str

        self.current_level = level
        level_graph = '  '.join(self.level_graph[:level])
        for i, c in enumerate(self.level_graph):
            if c == '`':
                self.level_graph[i] = ' '
        cname = self.cname_prefix + str(self.var_index)
        left_margin = "{}${}".format('' if level == 0 else '--', cname)
        self.var_index += 1
        if not isinstance(expr, str):
            show_func = self.get_show_func(expr_typed, expr_nodetype)
            if show_func is not None:
                item_show_info = show_func(expr_casted)

        if item_show_info is not None:
            print("{}{} ({}) {} {}".format(
                  level_graph, left_margin, expr_typed, expr, item_show_info))

        if not isinstance(expr, str):
            gdb_set_convenience_variable(cname, expr_casted)
            walk_func = self.get_walk_func(expr_typed, expr_nodetype)
            if walk_func is None:
                return
            children = walk_func(expr_casted)
            if not children:
                return

            if len(self.level_graph) < level + 1:
                self.level_graph.append('|')
            else:
                self.level_graph[level] = '|'

            for i, child in enumerate(children):
                if i == len(children) - 1:
                    self.level_graph[level] = '`'
                self.do_walk(child, level + 1)

    def get_action_func(self, item_type, action_prefix):
        def type_name(typ):
            return typ.name if typ.name != None and hasattr(typ, 'name') else str(typ)
        func_name = action_prefix + type_name(item_type)
        if hasattr(self, func_name):
            return getattr(self, func_name)

        for field in item_type.fields():
            if not field.is_base_class:
                continue
            typ = field.type
            func_name = action_prefix + type_name(typ)

            if hasattr(self, func_name):
                return getattr(self, func_name)

            return self.get_action_func(typ, action_prefix)
        return None

    def get_walk_func(self, item_type, item_type_templ):
        if item_type_templ != None:
            return self.get_action_func(item_type_templ.target(), 'walk_templ_')
        else:
            return self.get_action_func(item_type.target(), 'walk_')

    def get_show_func(self, item_type, item_type_templ):
        if item_type_templ != None:
            return self.get_action_func(item_type_templ.target(), 'show_templ_')
        else:
            return self.get_action_func(item_type.target(), 'show_')

class ItemDisplayer(object):
    """mysql item basic show functions"""

    def get_show_func(self, item_type, item_type_templ):
        if item_type_templ != None:
            return self.get_action_func(item_type_templ.target(), 'show_templ_')
        else:
            return self.get_action_func(item_type.target(), 'show_')

    def show_item_info(self, item):
        expr_typed = item.dynamic_type
        expr_casted = item.cast(expr_typed)
        expr_nodetype = None
        try:
            expr_nodetype = item.type.template_argument(0)
            if expr_nodetype.code != gdb.TYPE_CODE_PTR:
                expr_nodetype = item.type.template_argument(0).pointer()
        except (gdb.error, RuntimeError):
            expr_nodetype = None
            pass
        show_func = self.get_show_func(expr_typed, expr_nodetype)
        if show_func is not None:
            item_show_info = show_func(expr_casted)
        if item_show_info == None:
            return ''
        return item_show_info

    def show_Item_ident(self, item):
        db_cata = []
        if item['db_name']:
            db_cata.append(item['db_name'].string())
        if item['table_name']:
            db_cata.append(item['table_name'].string())
        if item['field_name']:
            db_cata.append(item['field_name'].string())
        return 'field = ' + '.'.join(db_cata)

    def show_Item_int(self, item):
        return 'value = ' + str(item['value'])

    show_Item_float = show_Item_int

    def show_Item_string(self, item):
        return 'value = ' + item['str_value']['m_ptr'].string()

    def show_Item_decimal(self, item):
        sym = gdb.lookup_global_symbol('Item_decimal::val_real()')
        result = sym.value()(item)
        return 'value = ' + str(result)

    def show_uchar(self, item):
        return item

    def show_Field(self, item):
        db_cata = []
        if item['table_name']:
            db_cata.append(item['table_name'].dereference().string())
        if item['field_name']:
            db_cata.append(item['field_name'].string())
        return 'field = ' + '.'.join(db_cata)

    def show_TABLE_LIST(self, val):
        alias = val['alias']
        db = val['db']
        join_cond = val['m_join_cond'] if val['m_join_cond'] != None else ''
        outer_cond = val['outer_join'] if val['outer_join'] != None else ''
        return '{}.{}'.format(db.string(), alias.string())

    def show_SARGABLE_PARAM(self, item):
        ''' {field = 0x7ff39d6efa58, arg_value = 0x7ff39e6af698, num_values = 1} '''
        info = 'field = ' + self.show_Field(item['field'])
        info += ', arg_value = \'' + self.show_item_info(item['arg_value'].dereference()) + '\''
        info += ', num_values = ' + str(item['num_values'])
        return info

    def show_Key_use(self, item):
        info = 'key = ' + str(item['key'])
        info += ', keypart = ' + str(item['keypart'])
        info += ', val = \'' + self.show_item_info(item['val']) + '\''
        info += ', null_rejecting = ' + str(item['null_rejecting'])
        info += ', used_tables = ' + str(item['used_tables'])
        info += ', table_ref = ' + self.show_TABLE_LIST(item['table_ref'])
        info += ', fanout = ' + str(item['fanout'])
        info += ', read_cost = ' + str(item['read_cost'])
        return info

class ExpressionTraverser(gdb.Command, TreeWalker, ItemDisplayer):
    """print mysql expression (Item) tree"""

    def __init__ (self):
        super(self.__class__, self).__init__("mysql exprtree", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql exprtree [Item]")
            return
        expr = gdb.parse_and_eval(arg)
        self.walk(expr)

    #
    # walk and show functions for each Item class
    #

    def walk_Item_func(self, val):
        children = []
        for i in range(val['arg_count']):
            children.append(val['args'][i])
        return children

    walk_Item_sum = walk_Item_func

    def walk_Item_cond(self, val):
        end_of_list = gdb.parse_and_eval('end_of_list').address
        item_list = val['list']
        nodetype = item_list.type.template_argument(0)
        cur_elt = item_list['first']
        children = []
        while cur_elt != end_of_list:
            info = cur_elt.dereference()['info']
            children.append(info.cast(nodetype.pointer()))
            cur_elt = cur_elt.dereference()['next']
        return children

    def walk_Item_equal(self, val):
        end_of_list = gdb.parse_and_eval('end_of_list').address
        item_list = val['fields']
        nodetype = item_list.type.template_argument(0)
        cur_elt = item_list['first']
        children = []
        children.append(val['const_item'])
        while cur_elt != end_of_list:
            info = cur_elt.dereference()['info']
            children.append(info.cast(nodetype.pointer()))
            cur_elt = cur_elt.dereference()['next']
        return children

ExpressionTraverser()

def print_TABLE_LIST(leaf_tables):
    tables = ''
    has_tables = False
    i = 0
    tl_cnname_prefix = get_convenince_name()
    while leaf_tables:
        has_tables = True
        lt = leaf_tables.dereference()
        table_name = lt['table_name'].string()
        table_name = table_name[-18:]
        if len(table_name) == 18:
            table_name = '...' + table_name[4:]

        tl_cnname = tl_cnname_prefix + str(i)
        i += 1
        gdb_set_convenience_variable(tl_cnname, lt)

        tables +=  '($' + tl_cnname + ')' + table_name + " " + lt['alias'].string() +  ", "
        leaf_tables = lt['next_leaf']

    if has_tables:
        tables = "tables: " + tables[0 : len(tables) - 2]
    else:
        tables = "no tables"
    return tables

def print_SELECT_LEX(select_lex):
    """print SELECT_LEX extra information"""

    leaf_tables = select_lex['leaf_tables']
    return print_TABLE_LIST(leaf_tables)

def print_SELECT_LEX_UNIT(select_lex_unit):
    try:
        return str("")
    except gdb.error:
        pass
    return ''

# For 8.0.25+
def print_Query_block(query_block):
    """print Query_block extra information"""

    leaf_tables = query_block['leaf_tables']
    return print_TABLE_LIST(leaf_tables)

def print_Query_expression(unit):
    try:
        return str("")
    except gdb.error:
        pass
    return ''

class QueryBlockTraverser(gdb.Command, TreeWalker):
    """print mysql query block tree"""
    def __init__ (self):
        super(self.__class__, self).__init__ ("mysql qbtree", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql qbtree [SELECT_LEX_UNIT/SELECT_LEX/Query_expression/Query_block]")
            return
        qb = gdb.parse_and_eval(arg)
        self.start_qb = qb.dereference()
        while qb.dereference()['master']:
            qb = qb.dereference()['master']

        self.walk(qb)

    def do_walk_query_block(self, val):
        blocks = []
        if not val['slave']:
            return blocks
        block = val['slave']
        blocks.append(block)
        while block['next']:
            block = block['next']
            blocks.append(block)
        return blocks
    
    walk_SELECT_LEX = do_walk_query_block
    walk_SELECT_LEX_UNIT = do_walk_query_block
    walk_Query_expression = do_walk_query_block
    walk_Query_block = do_walk_query_block

    def get_current_marker(self, val):
        if self.start_qb.address != val:
            return ''
        return ' <-'
    
    def show_SELECT_LEX(self, val):
        return print_SELECT_LEX(val) + self.get_current_marker(val)

    def show_SELECT_LEX_UNIT(self, val):
        return print_SELECT_LEX_UNIT(val) + self.get_current_marker(val)

    def show_Query_expression(self, val):
        return print_Query_expression(val) + self.get_current_marker(val)

    def show_Query_block(self, val):
        return print_Query_block(val) + self.get_current_marker(val)

QueryBlockTraverser()

class TABLE_LIST_traverser(gdb.Command):
    """print leaf_tables"""
    def __init__ (self):
        super (TABLE_LIST_traverser, self).__init__("mysql tl",
                                                    gdb.COMMAND_OBSCURE)
    def invoke(self, arg, from_tty):
        table_list = gdb.parse_and_eval(arg)
        print(print_TABLE_LIST(table_list))

TABLE_LIST_traverser()

class SEL_TREE_traverser(gdb.Command, TreeWalker, ItemDisplayer):
    NO_MIN_RANGE = 1
    NO_MAX_RANGE = 2
    NEAR_MIN = 4
    NEAR_MAX = 8
    """print SEL_TREE"""
    def __init__ (self):
        super (self.__class__, self).__init__("mysql sel_tree", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql sel_tree [SEL_TREE]")
            return
        sel_tree = gdb.parse_and_eval(arg)
        if sel_tree:
            self.walk(sel_tree)
        else:
            print('None')

    def walk_SEL_TREE(self, val):
        sel_tree_keys = val['keys']
        return self.sel_tree_keys_to_list(val)

    def show_SEL_TREE(self, val):
        sel_tree_keys = val['keys']
        sel_tree_type = val['type']
        return "[type={},keys.m_size={}]".format(sel_tree_type, sel_tree_keys['m_size'])

    def walk_SEL_ROOT(self, val):
        out_list = []
        if val:
            out_list.append(val['root'])
        return out_list

    def show_SEL_ROOT(self, val):
        if not val:
            return "None"
        sel_root_type = val['type']
        sel_root_use_count = val['use_count']
        sel_root_elements = val['elements']
        return "[type={}, use_count={}, elements={}]".format(sel_root_type, sel_root_use_count, sel_root_elements);

    def walk_SEL_ARG(self, val):
        sel_arg_field = val['field']
        if not sel_arg_field:
             return None
        return self.sel_arg_tree_to_list(val)

    def show_SEL_ARG(self, val):
        sel_arg_field = val['field']
        if not sel_arg_field:
             return None
        level_graph = '  '.join(self.level_graph[:self.current_level])
        for i, c in enumerate(self.level_graph):
            if c == '`':
                self.level_graph[i] = ' '
        left_margin = "  |"

        if len(self.level_graph) < self.current_level + 1:
            self.level_graph.append('|')
        else:
            self.level_graph[self.current_level] = '|'

        field_show_info = ''
        if val['field']:
            field_show_info = self.get_item_show_info(val['field'])
            field_show_info = "{}{} field = {}".format(level_graph, left_margin,
                                    field_show_info)
        
        sel_root_max_flag = val['max_flag']
        sel_root_min_flag = val['min_flag']
        left_parenthese = '['
        right_parenthese = ']'
        min_item_show_info = ''
        max_item_show_info = ''
        min_value = val['min_value']
        max_value = val['max_value']
        min_item = None
        max_item = None
        try:
            min_item = val['min_item']
            max_item = val['max_item']
        except (gdb.error, RuntimeError):
            min_item = None
            max_item = None

        if min_item and self.NO_MIN_RANGE & sel_root_min_flag == 0:
            min_item_show_info = self.get_item_show_info(val['min_item'])
            if self.NEAR_MIN & sel_root_min_flag > 0:
                left_parenthese = "("
        else:
            if min_item:
                min_item_show_info = " -infinity"
                left_parenthese = "("
            else:
                min_item_show_info = self.get_item_show_info(val['min_value'])
                if self.NEAR_MIN & sel_root_min_flag > 0:
                    left_parenthese = "("

        max_item_show_info = ''
        if max_item and self.NO_MAX_RANGE & sel_root_max_flag == 0:
            max_item_show_info = self.get_item_show_info(val['max_item'])
            if self.NEAR_MAX & sel_root_max_flag > 0:
                right_parenthese = ")"
        else:
            if max_item:
                max_item_show_info = " +infinity"
                right_parenthese = ")"
            else:
                max_item_show_info = self.get_item_show_info(val['max_value'])
                if self.NEAR_MAX & sel_root_max_flag > 0:
                    right_parenthese = ")"

        item_show_info = ''
        if sel_root_max_flag == 0 and sel_root_min_flag == 0 and min_value == max_value:
            item_show_info = "{}{} equal = {}{} {}".format(level_graph, left_margin, left_parenthese, min_item_show_info,
                                                           right_parenthese)
        else:
            item_show_info = "{}{} scope = {}{},{} {}".format(level_graph, left_margin, left_parenthese, min_item_show_info,
                                                              max_item_show_info, right_parenthese)
        return "[color={}, is_asc={}, minflag={}, maxflag={}, part={}]\n{}\n{}".format(
                     val['color'], val['is_ascending'], sel_root_min_flag,
                     sel_root_max_flag, val['part'], field_show_info, item_show_info)

    def get_item_show_info(self, expr):
        item_show_info = ''
        cname = self.cname_prefix + str(self.var_index)
        self.var_index += 1
        expr_typed = expr.dynamic_type
        expr_casted = expr.cast(expr_typed)
        item_show_info = " ${} ({}) {}".format(
                         cname, expr_typed, expr)
        show_func = self.get_show_func(expr_typed, None)
        if show_func is not None:
             item_show_info = "{} {}".format(item_show_info, show_func(expr_casted))
        return item_show_info

    def sel_tree_keys_to_list(self, val):
        out_list = []
        sel_tree_keys = val['keys']
        sel_tree_keys_array = sel_tree_keys['m_array']
        for i in range(sel_tree_keys['m_size']):
            out_list.append(sel_tree_keys_array[i])
        return out_list

    def sel_arg_tree_to_list(self, val):
        out_list = []
        sel_arg_left = val['left']
        if sel_arg_left:
            out_list.append(sel_arg_left)
        sel_arg_right = val['right']
        if sel_arg_right:
            out_list.append(sel_arg_right)
        sel_arg_next_part = val['next_key_part']
        if sel_arg_next_part:
            out_list.append(sel_arg_next_part)
        return out_list

SEL_TREE_traverser()

class JOIN_LIST_traverser(gdb.Command, TreeWalker, ItemDisplayer):
    """print join list TREE"""
    def __init__ (self):
        super (self.__class__, self).__init__("mysql join_tree", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql join_tree [join_list]")
            return
        join_list = gdb.parse_and_eval(arg)
        if join_list:
            self.walk(join_list)

    def walk_templ_TABLE_LIST(self, val):
        expr_typed = val.dynamic_type
        if (str(expr_typed) == 'memroot_deque<TABLE_LIST*>'):
            return std_deque_to_list(val)
        elif (str(expr_typed) == 'mem_root_deque<TABLE_LIST*>'):
            return mem_root_deque_to_list(val)

        nodetype = val.type.template_argument(0)
        cur_elt = val['first']
        end_of_list = gdb.parse_and_eval('end_of_list').address
        children = []
        while cur_elt != end_of_list:
            info = cur_elt.dereference()['info']
            nodeinfo = info.cast(nodetype.pointer())
            children.append(nodeinfo)
            cur_elt = cur_elt.dereference()['next']
        return children

    def walk_TABLE_LIST(self, val):
        alias = val['alias']
        children = []
        if alias != None and (gdb.Value(alias).string() == '(nest_last_join)' or gdb.Value(alias).string() == '(nested_join)' or 
            gdb.Value(alias).string() == '(sj-nest)'):
            nested_join = val['nested_join'].dereference()
            children.append(nested_join['join_list']) 
        return children

    def walk_NESTED_JOIN(self, val):
        join_list = val['join_list']
        children = []
        children.append(join_list)
        return children

    def show_JOIN(self, val):
        table_count = val['query_block']['leaf_table_count']
        sj_nests = val['query_block']['sj_nests']
        funcname = "(uint) (*("+str(sj_nests.type)+"*)("+str(sj_nests.address)+")).size()"
        sj_nests_count = gdb.parse_and_eval(funcname);
        win_count = 0
        if val['m_windows'] != None:
            win_count = val['m_windows']['elements']
        best_ref_count = table_count + sj_nests_count + 2 + win_count
        return 'tables={}, table_count={}, const_tables={}, primary_tables={}, tmp_tables={}, c_sj_nests={}, c_best_ref={}'.format(
            val['tables'], table_count, val['const_tables'], val['primary_tables'], val['tmp_tables'], sj_nests_count, best_ref_count)

    def walk_JOIN(self, val):
        join_tab = val['join_tab']
        expr_typed = join_tab.dynamic_type
        children = []
        i = 0
        table_count = val['query_block']['leaf_table_count']
        children.append("-------join_tab array---------")
        while i != table_count:
            children.append(join_tab[i].address)
            i = i+1
        children.append("-------best_ref array---------")
        sj_nests = val['query_block']['sj_nests']
        funcname = "(uint) (*("+str(sj_nests.type)+"*)("+str(sj_nests.address)+")).size()"
        sj_nests_count = gdb.parse_and_eval(funcname);
        win_count = 0
        if val['m_windows'] != None:
            win_count = val['m_windows']['elements']
        best_ref_count = table_count + sj_nests_count + 2 + win_count
        best_ref = val['best_ref']
        i = 0
        while i != best_ref_count:
            children.append(best_ref[i])
            i = i+1
        children.append(val['keyuse_array'].address)
        return children

    def show_Key_use_array(self, val):
        return "size={}".format(val['m_size'])

    def walk_Key_use_array(self, val):
        elements = val['m_array']
        children = []
        for i in range(val['m_size']):
            children.append(elements+i)
        return children

    def show_JOIN_TAB(self, val):
        return 'table_name={},found_records={},read_time={},dependent={},use_quick={},const_keys={}, skip_scan_keys={}, quick_order_tested={}'.format(val['table_ref']['alias'], val['found_records'], val['read_time'], val['dependent'], val['use_quick'], val['const_keys']['map'], val['skip_scan_keys']['map'], val['quick_order_tested']['map'])

JOIN_LIST_traverser()
#
# pretty printers
#

def get_value_from_list_node(nodetype, node, conname_prefix, index):
    """Returns the value held in an list_node<_Val>"""

    val = node['info'].cast(nodetype.pointer())
    val = val.cast(val.dynamic_type)

    conven_name = '%s%d' % (conname_prefix, index)
    gdb_set_convenience_variable(conven_name, val)

    return val

class PrinterIterator(object):
    """A helper class, compatiable with python 2.0"""
    def next(self):
        """For python 2"""
        return self.__next__()

class ListPrinter(object):
    """Print a MySQL List"""

    class _iterator(PrinterIterator):
        def __init__(self, nodetype, head):
            self.nodetype = nodetype
            self.base = head
            self.count = 0
            self.end_of_list = gdb.parse_and_eval('end_of_list').address
            self.convenience_name_prefix = get_convenince_name()

        def __iter__(self):
            return self

        def __next__(self):
            if self.base == self.end_of_list:
                raise StopIteration
            elt = self.base.dereference()
            self.base = elt['next']
            cname = '%s%d' % (self.convenience_name_prefix, self.count)
            val = get_value_from_list_node(self.nodetype, elt, self.convenience_name_prefix, self.count)
            self.count = self.count + 1
            gdb_set_convenience_variable(cname, elt)
            return ('$%s' % (cname), '(%s) %s' % (val.type, val))

    def __init__(self, val):
        self.typename = val.type
        self.val = val

    def children(self):
        nodetype = self.typename.template_argument(0)
        return self._iterator(nodetype, self.val['first'])

    def to_string(self):
        return '%s' % self.typename if self.val['elements'] != 0 else 'empty %s' % self.typename

def get_value_from_deque_node(nodetype, node, conname_prefix, index):
    """Returns the value held in an list_node<_Val>"""

    val = node['info'].cast(nodetype.pointer())
    val = val.cast(val.dynamic_type)

    conven_name = '%s%d' % (conname_prefix, index)
    gdb_set_convenience_variable(conven_name, val)

    return val

class memroot_dequePrinter(object):
    """Print a MySQL memroot_deque List"""

    class _iterator(PrinterIterator):
        def __init__(self, nodetype, head):
            self.nodetype = nodetype
            self.base = head['_M_impl']['_M_start']['_M_cur']
            self.end = head['_M_impl']['_M_start']['_M_last']
            self.node = head['_M_impl']['_M_start']['_M_node']
            self.last= head['_M_impl']['_M_finish']['_M_cur']
            self.count = 0
            self.convenience_name_prefix = get_convenince_name()
            self.buffer_size = 1
            if nodetype.sizeof < 512:
                self.buffer_size = int (512 / nodetype.sizeof)

        def __iter__(self):
            return self

        def __next__(self):
            if self.base == self.last:
                raise StopIteration
            elt = self.base.dereference()
            cname = '%s%d' % (self.convenience_name_prefix, self.count)
            self.count = self.count + 1
            self.base = self.base + 1
            if self.base == self.end: 
                self.node = self.node + 1
                self.base = self.node[0]
                self.end = self.base + self.buffer_size 
            gdb_set_convenience_variable(cname, elt)
            return ('$%s' % (cname), '(%s) %s' % (elt.type, elt))

    def __init__(self, val):
        self.typename = val.type
        self.val = val

    def children(self):
        nodetype = self.typename.template_argument(0)
        return self._iterator(nodetype, self.val)

    def to_string(self):
        return '%s' % self.typename

# mem_root_deque is from 8.0.22+
class mem_root_dequePrinter(object):
    """Print a MySQL mem_root_deque List"""

    class _iterator(PrinterIterator):
        def __init__(self, nodetype, head):
            self.nodetype = nodetype
            self.base = head['m_begin_idx']
            self.end = head['m_end_idx']
            self.elements = head['block_elements']
            self.blocks = head['m_blocks']
            self.count = 0
            self.convenience_name_prefix = get_convenince_name()

        def __iter__(self):
            return self

        def __next__(self):
            if self.base == self.end:
                raise StopIteration
            elt = self.blocks[self.base / self.elements]['elements'][self.base % self.elements]
            cname = '%s%d' % (self.convenience_name_prefix, self.count)
            self.count = self.count + 1
            self.base = self.base + 1
            gdb_set_convenience_variable(cname, elt)
            return ('$%s' % (cname), '(%s) %s' % (elt.type, elt))

    def __init__(self, val):
        self.typename = val.type
        self.val = val

    def children(self):
        nodetype = self.typename.template_argument(0)
        return self._iterator(nodetype, self.val)

    def to_string(self):
        return '%s' % self.typename

class Mem_root_arrayPrinter(object):
    """Print a MySQL Mem_root_array List"""

    class _iterator(PrinterIterator):
        def __init__(self, nodetype, head):
            self.nodetype = nodetype
            self.elements = head['m_array']
            self.end = head['m_size']
            self.count = 0
            self.base = 0
            self.convenience_name_prefix = get_convenince_name()

        def __iter__(self):
            return self

        def __next__(self):
            if self.base == self.end:
                raise StopIteration
            elt = self.elements[self.base]
            cname = '%s%d' % (self.convenience_name_prefix, self.count)
            self.count = self.count + 1
            self.base = self.base + 1
            gdb_set_convenience_variable(cname, elt)
            return ('$%s' % (cname), '(%s) %s' % (elt.type, elt))

    def __init__(self, val):
        self.typename = val.type
        self.val = val

    def children(self):
        nodetype = self.typename.template_argument(0)
        return self._iterator(nodetype, self.val)

    def to_string(self):
        return '%s' % self.typename

import gdb.printing
def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter(
        "mysqld")
    pp.add_printer('List', '^List<.*>$', ListPrinter)
    pp.add_printer('memroot_deque', '^memroot_deque<.*>$', memroot_dequePrinter)
    pp.add_printer('mem_root_deque', '^mem_root_deque<.*>$', mem_root_dequePrinter)
    pp.add_printer('Mem_root_array', '^Mem_root_array<.*>$', Mem_root_arrayPrinter)
    return pp

gdb.printing.register_pretty_printer(
    gdb.current_objfile(),
    build_pretty_printer(),
    True)
