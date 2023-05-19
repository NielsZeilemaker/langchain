"""SQLAlchemy wrapper around a database."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, cast

import sqlalchemy
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable

from langchain import utils


def _format_index(index: sqlalchemy.engine.interfaces.ReflectedIndex) -> str:
    return (
        f'Name: {index["name"]}, Unique: {index["unique"]},'
        f' Columns: {str(index["column_names"])}'
    )


class SQLLikeDatabase(ABC):
    """Abstract class for SQLLike sources"""

    def __init__(
        self,
        sample_rows_in_table_info: int,
        indexes_in_table_info: bool,
        include_tables: Optional[List[str]],
        ignore_tables: Optional[List[str]],
        custom_table_info: Optional[Dict[str, str]],
        view_support: bool,
    ):
        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._indexes_in_table_info = indexes_in_table_info
        self._view_support = view_support

        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._include_tables = set(include_tables or [])
        self._ignore_tables = set(ignore_tables or [])
        self._custom_table_info = custom_table_info

        self._all_tables = set(self.get_all_table_names())
        self._usable_tables = set(self.get_usable_table_names()) or self._all_tables


        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )

        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )

        if self._custom_table_info:
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table])
                for table in self._custom_table_info
                if table in intersection
            )

    @abstractmethod
    def get_all_table_names(self) -> Iterable[str]:
        """Get names of all tables"""
        pass

    @abstractmethod
    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        pass

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    @abstractmethod
    def _get_create_table(self, table_name: str) -> str:
        pass

    def _get_custom_table_info(self, table_name: str) -> Optional[str]:
        return self._custom_table_info.get(table_name) if self._custom_table_info else None

    @abstractmethod
    def _get_table_indexes(self, table_name: str) -> str:
        pass

    @abstractmethod
    def _get_columns(self, table_name: str) -> str:
        pass

    @abstractmethod
    def _get_sample_rows(self, table_name: str) -> str:
        pass

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        tables = []
        for table in all_table_names:
            custom_table_info = self._get_custom_table_info(table)
            if custom_table_info:
                # if we have custom info, use that
                tables.append(custom_table_info)

            else:
                # else, add create table command
                table_info = self._get_create_table(table)
                if self._indexes_in_table_info or self._sample_rows_in_table_info:
                    table_info += "\n\n/*"
                    if self._indexes_in_table_info:
                        table_info += f"\nTable Indexes:"
                        table_info += f"\n{self._get_table_indexes(table)}\n"
                    if self._sample_rows_in_table_info:
                        table_info += f"\n{self._sample_rows_in_table_info} rows from {table} table:"
                        table_info += f"\n{self._get_columns(table)}"
                        table_info += f"\n{self._get_sample_rows(table)}\n"
                    table_info += "*/"
                tables.append(table_info)
        final_str = "\n\n".join(tables)
        return final_str

    def get_table_info_no_throw(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return self.get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    @abstractmethod
    def run(self, command: str, fetch: str = "all") -> str:
        pass

    def run_no_throw(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(command, fetch)
        except SQLAlchemyError as e:
            """Format the error message"""
            return f"Error: {e}"


class SQLDatabase(SQLLikeDatabase):
    """SQLAlchemy wrapper around a database."""

    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        metadata: Optional[MetaData] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        indexes_in_table_info: bool = False,
        custom_table_info: Optional[dict] = None,
        view_support: bool = False,
    ):
        """Create engine from database URI."""
        self._engine = engine
        self._schema = schema
        self._inspector = inspect(self._engine)

        super().__init__(
            ignore_tables=ignore_tables,
            include_tables=include_tables,
            sample_rows_in_table_info=sample_rows_in_table_info,
            indexes_in_table_info=indexes_in_table_info,
            custom_table_info=custom_table_info,
            view_support=view_support,
        )

        self._metadata = metadata or MetaData()
        self._metadata.reflect(
            views=view_support,
            bind=self._engine,
            only=list(self._usable_tables),
            schema=self._schema,
        )

    @classmethod
    def from_uri(
        cls, database_uri: str, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> SQLDatabase:
        """Construct a SQLAlchemy engine from URI."""
        _engine_args = engine_args or {}
        return cls(create_engine(database_uri, **_engine_args), **kwargs)

    @classmethod
    def from_databricks(
        cls,
        catalog: str,
        schema: str,
        host: Optional[str] = None,
        api_token: Optional[str] = None,
        warehouse_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> SQLDatabase:
        """
        Class method to create an SQLDatabase instance from a Databricks connection.
        This method requires the 'databricks-sql-connector' package. If not installed,
        it can be added using `pip install databricks-sql-connector`.

        Args:
            catalog (str): The catalog name in the Databricks database.
            schema (str): The schema name in the catalog.
            host (Optional[str]): The Databricks workspace hostname, excluding
                'https://' part. If not provided, it attempts to fetch from the
                environment variable 'DATABRICKS_HOST'. If still unavailable and if
                running in a Databricks notebook, it defaults to the current workspace
                hostname. Defaults to None.
            api_token (Optional[str]): The Databricks personal access token for
                accessing the Databricks SQL warehouse or the cluster. If not provided,
                it attempts to fetch from 'DATABRICKS_API_TOKEN'. If still unavailable
                and running in a Databricks notebook, a temporary token for the current
                user is generated. Defaults to None.
            warehouse_id (Optional[str]): The warehouse ID in the Databricks SQL. If
                provided, the method configures the connection to use this warehouse.
                Cannot be used with 'cluster_id'. Defaults to None.
            cluster_id (Optional[str]): The cluster ID in the Databricks Runtime. If
                provided, the method configures the connection to use this cluster.
                Cannot be used with 'warehouse_id'. If running in a Databricks notebook
                and both 'warehouse_id' and 'cluster_id' are None, it uses the ID of the
                cluster the notebook is attached to. Defaults to None.
            engine_args (Optional[dict]): The arguments to be used when connecting
                Databricks. Defaults to None.
            **kwargs (Any): Additional keyword arguments for the `from_uri` method.

        Returns:
            SQLDatabase: An instance of SQLDatabase configured with the provided
                Databricks connection details.

        Raises:
            ValueError: If 'databricks-sql-connector' is not found, or if both
                'warehouse_id' and 'cluster_id' are provided, or if neither
                'warehouse_id' nor 'cluster_id' are provided and it's not executing
                inside a Databricks notebook.
        """
        try:
            from databricks import sql  # noqa: F401
        except ImportError:
            raise ValueError(
                "databricks-sql-connector package not found, please install with"
                " `pip install databricks-sql-connector`"
            )
        context = None
        try:
            from dbruntime.databricks_repl_context import get_context

            context = get_context()
        except ImportError:
            pass

        default_host = context.browserHostName if context else None
        if host is None:
            host = utils.get_from_env("host", "DATABRICKS_HOST", default_host)

        default_api_token = context.apiToken if context else None
        if api_token is None:
            api_token = utils.get_from_env(
                "api_token", "DATABRICKS_API_TOKEN", default_api_token
            )

        if warehouse_id is None and cluster_id is None:
            if context:
                cluster_id = context.clusterId
            else:
                raise ValueError(
                    "Need to provide either 'warehouse_id' or 'cluster_id'."
                )

        if warehouse_id and cluster_id:
            raise ValueError("Can't have both 'warehouse_id' or 'cluster_id'.")

        if warehouse_id:
            http_path = f"/sql/1.0/warehouses/{warehouse_id}"
        else:
            http_path = f"/sql/protocolv1/o/0/{cluster_id}"

        uri = (
            f"databricks://token:{api_token}@{host}?"
            f"http_path={http_path}&catalog={catalog}&schema={schema}"
        )
        return cls.from_uri(database_uri=uri, engine_args=engine_args, **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        warnings.warn(
            "This method is deprecated - please use `get_usable_table_names`."
        )
        return self.get_usable_table_names()

    def get_all_table_names(self) -> Iterable[str]:
        """Get names of all tables."""
        self._all_tables = set(self._inspector.get_table_names(schema=self._schema))
        if self._view_support:
            self._all_tables.update(self._inspector.get_view_names(schema=self._schema))
        return self._all_tables

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables

        usable_tables = self._all_tables - self._ignore_tables

        if self.dialect == "sqlite":
            return filter(
                lambda table_name: not table_name.startswith("sqlite_"), usable_tables
            )
        return usable_tables

    def __get_table(self, table_name: str) -> Optional[Table]:
        for table in self._metadata.sorted_tables:
            if table.name == table_name:
                return table
        return None

    def _get_create_table(self, table_name: str) -> str:
        table = cast(Table, self.__get_table(table_name))
        create_table = str(CreateTable(table).compile(self._engine))
        return create_table.rstrip()

    def _get_table_indexes(self, table_name: str) -> str:
        indexes = self._inspector.get_indexes(table_name)
        return "\n".join(map(_format_index, indexes))

    def _get_columns(self, table_name: str) -> str:
        table = cast(Table, self.__get_table(table_name))
        return "\t".join([col.name for col in table.columns])

    def _get_sample_rows(self, table_name: str) -> str:
        table = cast(Table, self.__get_table(table_name))

        # build the select command
        command = select(table).limit(self._sample_rows_in_table_info)

        try:
            # get the sample rows
            with self._engine.connect() as connection:
                sample_rows_result = connection.execute(command)  # type: ignore
                # shorten values in the sample rows
                sample_rows = list(
                    map(lambda ls: [str(i)[:100] for i in ls], sample_rows_result)
                )

            # save the sample rows in string format
            return "\n".join(["\t".join(row) for row in sample_rows])

        # in some dialects when there are no rows in the table a
        # 'ProgrammingError' is returned
        except ProgrammingError:
            return ""

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            if self._schema is not None:
                if self.dialect == "snowflake":
                    connection.exec_driver_sql(
                        f"ALTER SESSION SET search_path='{self._schema}'"
                    )
                else:
                    connection.exec_driver_sql(f"SET search_path TO {self._schema}")
            cursor = connection.execute(text(command))
            if cursor.returns_rows:
                if fetch == "all":
                    result = cursor.fetchall()
                elif fetch == "one":
                    result = cursor.fetchone()[0]  # type: ignore
                else:
                    raise ValueError("Fetch parameter must be either 'one' or 'all'")
                return str(result)
        return ""
