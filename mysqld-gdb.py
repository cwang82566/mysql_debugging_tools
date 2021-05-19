from __future__ import print_function # python2.X support
import re

# some utilities
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

def gdb_set_convenience_variable(var_name, var):
    """set convenience variable in python script is supported by gdb 8.0
or above, this for lower gdb version support

    """
    pass

def get_convenince_name():
    return ''

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


#
# Some convenience variables for easy debug because they are macros
#
gdb_set_convenience_variable('MAX_TABLES', gdb.parse_and_eval('sizeof(uint64_t) * 8 - 3'))
gdb_set_convenience_variable('INNER_TABLE_BIT', gdb.parse_and_eval('((uint64_t)1) << ($MAX_TABLES + 0)'))
gdb_set_convenience_variable('OUTER_REF_TABLE_BIT', gdb.parse_and_eval('((uint64_t)1) << ($MAX_TABLES + 1)'))
gdb_set_convenience_variable('RAND_TABLE_BIT', gdb.parse_and_eval('((uint64_t)1) << ($MAX_TABLES + 2)'))
gdb_set_convenience_variable('PSEUDO_TABLE_BITS', gdb.parse_and_eval('($INNER_TABLE_BIT | $OUTER_REF_TABLE_BIT | $RAND_TABLE_BIT)'))

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
        expr_typed = expr.dynamic_type
        expr_casted = expr.cast(expr_typed)
        self.current_level = level
        level_graph = '  '.join(self.level_graph[:level])
        for i, c in enumerate(self.level_graph):
            if c == '`':
                self.level_graph[i] = ' '
        cname = self.cname_prefix + str(self.var_index)
        left_margin = "{}${}".format('' if level == 0 else '--', cname)
        self.var_index += 1
        item_show_info = ''
        show_func = self.get_show_func(expr_typed.target())
        if show_func is not None:
            item_show_info = show_func(expr_casted)
        if item_show_info is not None:
            print("{}{} ({}) {} {}".format(
                  level_graph, left_margin, expr_typed, expr, item_show_info))
        gdb_set_convenience_variable(cname, expr_casted)
        walk_func = self.get_walk_func(expr_typed.target())
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
            return typ.name if hasattr(typ, 'name') else str(typ)

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

    def get_walk_func(self, item_type):
        return self.get_action_func(item_type, 'walk_')

    def get_show_func(self, item_type):
        return self.get_action_func(item_type, 'show_')

# Define a mysql command prefix for all mysql related command
gdb.Command('mysql', gdb.COMMAND_DATA, prefix=True)

class ItemDisplayer(object):
    """mysql item basic show functions"""
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
        
class ExpressionTraverser(gdb.Command, TreeWalker, ItemUtility):
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

class QueryBlockTraverser(gdb.Command, TreeWalker):
    """print mysql query block tree"""
    def __init__ (self):
        super(self.__class__, self).__init__ ("mysql qbtree", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql qbtree [SELECT_LEX_UNIT/SELECT_LEX]")
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

    def get_current_marker(self, val):
        if self.start_qb.address != val:
            return ''
        return ' <-'
    
    def show_SELECT_LEX(self, val):
        return print_SELECT_LEX(val) + self.get_current_marker(val)

    def show_SELECT_LEX_UNIT(self, val):
        return print_SELECT_LEX_UNIT(val) + self.get_current_marker(val)


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

class SEL_TREE_traverser(gdb.Command, TreeWalker, ItemUtility):
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
        if val['field_item']:
            field_show_info = self.get_item_show_info(val['field_item'])
            field_show_info = "{}{} field = {}".format(level_graph, left_margin,
                                    field_show_info)
        
        sel_root_max_flag = val['max_flag']
        sel_root_min_flag = val['min_flag']
        left_parenthese = '['
        right_parenthese = ']'
        min_item_show_info = ''
        if val['min_item'] and self.NO_MIN_RANGE & sel_root_min_flag == 0:
            min_item_show_info = self.get_item_show_info(val['min_item'])
            if self.NEAR_MIN & sel_root_min_flag > 0:
                left_parenthese = "("
        else:
            min_item_show_info = " -infinity"
            left_parenthese = "("

        max_item_show_info = ''
        if val['max_item'] and self.NO_MAX_RANGE & sel_root_max_flag == 0:
            max_item_show_info = self.get_item_show_info(val['max_item'])
            if self.NEAR_MAX & sel_root_max_flag > 0:
                right_parenthese = ")"
        else:
            max_item_show_info = " +infinity"
            right_parenthese = ")"

        item_show_info = ''
        if sel_root_max_flag == 0 and sel_root_min_flag == 0 and val['max_item'] == val['min_item']:
            item_show_info = "{}{} equal = {}{} {}".format(level_graph, left_margin, left_parenthese, min_item_show_info,
                                                      right_parenthese)
        else:
            item_show_info = "{}{} scope = {}{},{} {}".format(level_graph, left_margin, left_parenthese, min_item_show_info,
                                                      max_item_show_info, right_parenthese)
        return "[color={}, is_asc={}, minflag={}, maxflag={}, part={}, selectivity={}]\n{}\n{}".format(
                     val['color'], val['is_ascending'], sel_root_min_flag,
                     sel_root_max_flag, val['part'], val['selectivity'], field_show_info, item_show_info)

    def get_item_show_info(self, expr):
        item_show_info = ''
        cname = self.cname_prefix + str(self.var_index)
        self.var_index += 1
        expr_typed = expr.dynamic_type
        expr_casted = expr.cast(expr_typed)
        item_show_info = " ${} ({}) {}".format(
                         cname, expr_typed, expr)
        show_func = self.get_show_func(expr_typed.target())
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
            count = self.count
            self.count = self.count + 1
            val = get_value_from_list_node(self.nodetype, elt, self.convenience_name_prefix, count)
            return ('%s[%d]' % (self.convenience_name_prefix, count), '(%s) %s' % (val.type, val))

    def __init__(self, val):
        self.typename = val.type
        self.val = val

    def children(self):
        nodetype = self.typename.template_argument(0)
        return self._iterator(nodetype, self.val['first'])

    def to_string(self):
        return '%s' % self.typename if self.val['elements'] != 0 else 'empty %s' % self.typename

import gdb.printing
def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter(
        "mysqld")
    pp.add_printer('List', '^List<.*>$', ListPrinter)
    return pp

gdb.printing.register_pretty_printer(
    gdb.current_objfile(),
    build_pretty_printer(),
    True)
